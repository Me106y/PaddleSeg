import os
import yaml
import paddle
import numpy as np

# 修复：使用PaddleSeg的组件
from paddleseg.cvlibs import manager, Config, SegBuilder
from paddleseg.utils import utils


def export_inference_model(config_path, weight_path, save_dir, input_shape=[1, 3, 640, 640]):
    """
    加载配置和权重，导出Paddle静态图推理模型（.pdmodel + .pdiparams）

    参数:
        config_path (str): 配置文件路径
        weight_path (str): 权重文件路径 (.pdparams)
        save_dir (str): 推理模型保存目录
        input_shape (list): 输入张量形状，默认 [1, 3, 640, 640]
    """

    # 1. 解析路径
    config_path = os.path.abspath(config_path)
    weight_path = os.path.abspath(weight_path)
    save_dir = os.path.abspath(save_dir)

    print("🔧 使用配置文件:", config_path)
    print("🔧 使用权重文件:", weight_path)
    print("📁 模型将保存到:", save_dir)

    # 2. 加载配置文件 (使用PaddleSeg的Config类)
    print("1. 加载配置文件...")
    cfg = Config(config_path)

    # 3. 构建模型结构 (使用SegBuilder)
    print("2. 构建模型结构...")
    builder = SegBuilder(cfg)
    model = builder.model

    # 4. 加载权重
    print("3. 加载权重文件...")
    if os.path.exists(weight_path):
        utils.load_entire_model(model, weight_path)
        print("✅ 权重加载成功")
    else:
        print(f"❌ 权重文件不存在: {weight_path}")
        raise FileNotFoundError(f"权重文件未找到: {weight_path}")

    # 5. 设置为评估模式
    model.eval()

    # 6. 验证模型是否可以前向推理
    print("4. 验证模型前向推理能力...")
    dummy_input = paddle.randn(input_shape, dtype="float32")
    try:
        output = model(dummy_input)
        # PaddleSeg模型输出通常是列表
        if isinstance(output, list):
            for i, out in enumerate(output):
                print(f"✅ 前向推理测试通过，输出 {i} 形状: {out.shape}")
        else:
            print(f"✅ 前向推理测试通过，输出形状: {output.shape}")
    except Exception as e:
        print(f"❌ 前向推理失败: {e}")
        raise

    # 7. 定义输入规格
    input_spec = paddle.static.InputSpec(
        shape=input_shape,
        dtype="float32",
        name="x"
    )

    # 8. 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "inference")

    # 9. 导出模型
    print("5. 导出推理模型...")
    paddle.jit.save(
        model,
        path=save_path,
        input_spec=[input_spec]
    )

    print(f"✅ 模型已成功导出到: {save_dir}")
    print("生成文件：", os.listdir(save_dir))


if __name__ == "__main__":
    # 配置参数
    # CONFIG_PATH = "C:\\Users\\16782\\Desktop\\PaddleSeg\\configs\\quick_start\\repvit_v8_14.yml"  # 示例配置
    # WEIGHT_PATH = "C:\\Users\\16782\\Desktop\\PaddleSeg\\output\\repvit_v8_14\\best_model\\model.pdparams" # 示例权重
    # SAVE_DIR = "C:\\Users\\16782\\Desktop\\PaddleSeg\\output\\repvit_v8_14\\inference_model"
    CONFIG_PATH = "C:\\Users\\16782\\Desktop\\PaddleSeg\\configs\\quick_start\\portrait_pp_humansegv2_lite.yml"  # 示例配置
    WEIGHT_PATH = "C:\\Users\\16782\\Desktop\\PaddleSeg\\output\\test\\model.pdparams" # 示例权重
    SAVE_DIR = "C:\\Users\\16782\\Desktop\\PaddleSeg\\output\\test\\inference_model"
    INPUT_SHAPE = [1, 3, 256, 144]  # 根据你的模型调整

    # 执行导出
    try:
        export_inference_model(CONFIG_PATH, WEIGHT_PATH, SAVE_DIR, INPUT_SHAPE)
    except Exception as e:
        print(f"\n导出过程中发生错误: {e}")