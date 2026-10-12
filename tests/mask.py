import cv2
import numpy as np
import paddle.inference as paddle_infer
import os

def predict_and_overlay(img_path, model_dir, save_path):
    # 1. 创建预测器
    config = paddle_infer.Config(
        os.path.join(model_dir, "model.pdmodel"),
        os.path.join(model_dir, "model.pdiparams")
    )
    config.enable_use_gpu(100, 0)
    config.disable_glog_info()
    config.enable_memory_optim()
    predictor = paddle_infer.create_predictor(config)

    # 2. 获取输入句柄
    input_names = predictor.get_input_names()
    input_handle = predictor.get_input_handle(input_names[0])

    # 3. 图像预处理
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not open or find the image at {img_path}")
        return
    original_h, original_w = img.shape[:2]
    target_h, target_w = 144, 256  # 根据模型配置调整
    img_resized = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    img_arr = img_resized.astype('float32') / 255.0
    img_arr = img_arr.transpose((2, 0, 1))  # HWC to CHW
    img_arr = np.expand_dims(img_arr, axis=0)  # Add batch dimension
    print(f"input shape: {img_arr.shape}")

    # 4. 设置输入
    input_handle.copy_from_cpu(img_arr)

    # 5. 运行预测
    predictor.run()

    # 6. 获取输出
    output_names = predictor.get_output_names()
    output_handle = predictor.get_output_handle(output_names[0])
    result = output_handle.copy_to_cpu()
    print(f"output shape: {result.shape}")
    mask = result[0]
    mask = (mask > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    # 将mask转换为红色3通道掩码
    mask_3_channel = np.zeros_like(img)
    mask_3_channel[:, :, 2] = mask  # OpenCV的BGR格式，红色在最后一个通道

    # 创建红色掩码图像
    overlay = img.copy()
    alpha = 0.6  # 红色掩码透明度，可调整
    # 创建红色掩码
    red_mask = np.zeros_like(img)
    red_mask[:, :, 2] = 255  # BGR格式，红色通道
    # 只在人像区域融合红色掩码和原图
    overlay[mask == 255] = cv2.addWeighted(img[mask == 255], 1 - alpha, red_mask[mask == 255], alpha, 0)
    cv2.imwrite(save_path, overlay)
    print(f"Overlay image saved to {save_path}")

if __name__ == "__main__":
    model_dir = '../output/repvit_v8_14/inference_model'
    img_path = '../data/test4.jpg'
    save_path = './output_overlay.jpg'
    predict_and_overlay(img_path, model_dir, save_path)