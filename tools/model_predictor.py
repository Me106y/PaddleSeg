import time
import numpy as np
import paddle
import paddle.nn as nn
from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.utils import utils
import yaml
import os
from typing import Tuple, List, Dict, Optional
import cv2
from PIL import Image


class ModelPredictor:
    """模型预测器，支持预热和性能测试"""

    def __init__(self, config_path: str, model_path: str, device: str = 'auto'):
        """
        初始化模型预测器
    
        Args:
            config_path (str): 配置文件路径
            model_path (str): 模型参数文件路径
            device (str): 设备类型 ('gpu', 'cpu', 或 'auto')
        """
        self.config_path = config_path
        self.model_path = model_path
        
        # 自动检测设备
        if device == 'auto':
            try:
                paddle.set_device('gpu')
                self.device = 'gpu'
                print("检测到GPU支持，使用GPU")
            except Exception as e:
                paddle.set_device('cpu')
                self.device = 'cpu'
                print(f"GPU不可用({str(e)})，使用CPU")
        else:
            self.device = device
            paddle.set_device(device)
    
        # 加载配置和模型
        self.cfg = Config(config_path)
        self.builder = SegBuilder(self.cfg)
        self.model = self.builder.model
        self._load_pretrained_model()
    
        # 设置为评估模式
        self.model.eval()
    
        print(f"模型加载完成，使用设备: {self.device}")
        print(f"模型参数量: {self._count_parameters():,}")

    def _load_pretrained_model(self):
        """加载预训练模型"""
        if os.path.exists(self.model_path):
            utils.load_entire_model(self.model, self.model_path)
            print(f"成功加载模型: {self.model_path}")
            
            # 检查并切换RepViT模型到deploy模式以获得最佳推理性能
            self._switch_to_deploy_if_repvit()
        else:
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
    
    def _switch_to_deploy_if_repvit(self):
        """如果是RepViT模型，切换到deploy模式以获得最佳推理性能"""
        try:
            # 检查backbone是否有switch_to_deploy方法
            if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'switch_to_deploy'):
                self.model.backbone.switch_to_deploy()
                print("✓ RepViT模型已切换到deploy模式，推理性能已优化")
                return True
            
            # 检查模型本身是否有switch_to_deploy方法
            elif hasattr(self.model, 'switch_to_deploy'):
                self.model.switch_to_deploy()
                print("✓ RepViT模型已切换到deploy模式，推理性能已优化")
                return True
            
            # 递归检查所有子模块
            else:
                deployed = False
                for name, module in self.model.named_sublayers():
                    if hasattr(module, 'switch_to_deploy'):
                        module.switch_to_deploy()
                        print(f"✓ {name} 模块已切换到deploy模式")
                        deployed = True
                
                if deployed:
                    print("✓ RepViT相关模块已切换到deploy模式，推理性能已优化")
                    return True
                else:
                    print("ℹ 当前模型不是RepViT架构，无需切换deploy模式")
                    return False
                    
        except Exception as e:
            print(f"⚠ 切换deploy模式时出现警告: {str(e)}")
            print("模型将继续使用训练模式结构进行推理")
            return False

    def _count_parameters(self) -> int:
        """统计模型参数量"""
        return sum(p.numel() for p in self.model.parameters())

    def _preprocess_image(self, image_path: str, target_size: Tuple[int, int]) -> paddle.Tensor:
        """
        预处理图像

        Args:
            image_path (str): 图像路径
            target_size (Tuple[int, int]): 目标尺寸 (height, width)

        Returns:
            paddle.Tensor: 预处理后的图像张量
        """
        # 读取图像
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path  # 假设已经是numpy数组

        # 调整尺寸
        height, width = target_size
        image = cv2.resize(image, (width, height))

        # 归一化
        image = image.astype(np.float32) / 255.0

        # 标准化 (使用ImageNet标准)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        # 转换为CHW格式并添加batch维度
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)

        # 转换为paddle张量
        return paddle.to_tensor(image, dtype='float32')

    def _generate_dummy_input(self, batch_size: int, height: int, width: int, channels: int = 3) -> paddle.Tensor:
        """
        生成虚拟输入数据

        Args:
            batch_size (int): 批次大小
            height (int): 图像高度
            width (int): 图像宽度
            channels (int): 通道数

        Returns:
            paddle.Tensor: 虚拟输入张量
        """
        return paddle.randn([batch_size, channels, height, width], dtype='float32')

    def warmup(self, input_tensor: paddle.Tensor, warmup_runs: int = 10) -> None:
        """
        模型预热

        Args:
            input_tensor (paddle.Tensor): 输入张量
            warmup_runs (int): 预热运行次数
        """
        print(f"开始预热，运行 {warmup_runs} 次...")

        with paddle.no_grad():
            for i in range(warmup_runs):
                _ = self.model(input_tensor)
                if (i + 1) % 5 == 0:
                    print(f"预热进度: {i + 1}/{warmup_runs}")

        # 清理显存
        if self.device == 'gpu':
            paddle.device.cuda.empty_cache()

        print("预热完成")

    def predict_with_batch_timing(self,
                                  input_data,
                                  target_size: Tuple[int, int],
                                  total_runs: int = 5000,
                                  batch_size: int = 500,
                                  warmup_per_batch: int = 50,
                                  wait_between_batches: int = 10,
                                  return_predictions: bool = False) -> Dict:
        """
        分批次进行预测测试，每批次间有等待时间

        Args:
            input_data: 输入数据，可以是图像路径、numpy数组或paddle张量
            target_size (Tuple[int, int]): 目标尺寸 (height, width)
            total_runs (int): 总测试次数
            batch_size (int): 每批次测试次数
            warmup_per_batch (int): 每批次预热次数
            wait_between_batches (int): 批次间等待时间（秒）
            return_predictions (bool): 是否返回预测结果

        Returns:
            Dict: 包含预测结果和性能统计的字典
        """
        # 准备输入张量
        if isinstance(input_data, str):  # 图像路径
            input_tensor = self._preprocess_image(input_data, target_size)
        elif isinstance(input_data, np.ndarray):  # numpy数组
            input_tensor = self._preprocess_image(input_data, target_size)
        elif isinstance(input_data, paddle.Tensor):  # paddle张量
            input_tensor = input_data
        elif input_data is None:  # 生成虚拟数据
            input_tensor = self._generate_dummy_input(1, target_size[0], target_size[1])
        else:
            raise ValueError(f"不支持的输入数据类型: {type(input_data)}")

        print(f"输入尺寸: {list(input_tensor.shape)}")
        print(f"目标尺寸: {target_size}")
        print(f"总测试次数: {total_runs}")
        print(f"批次大小: {batch_size}")
        print(f"每批次预热: {warmup_per_batch}")
        print(f"批次间等待: {wait_between_batches}秒")

        # 计算批次数量
        num_batches = (total_runs + batch_size - 1) // batch_size
        print(f"总批次数: {num_batches}")

        all_times = []
        all_predictions = []
        batch_stats = []

        for batch_idx in range(num_batches):
            # 计算当前批次的实际运行次数
            current_batch_size = min(batch_size, total_runs - batch_idx * batch_size)
            
            print(f"\n=== 批次 {batch_idx + 1}/{num_batches} ====")
            print(f"当前批次测试次数: {current_batch_size}")

            # 每批次预热
            self.warmup(input_tensor, warmup_per_batch)

            # 当前批次的测试
            batch_times = []
            batch_predictions = []

            print(f"开始批次 {batch_idx + 1} 测试...")
            
            with paddle.no_grad():
                for i in range(current_batch_size):
                    # 同步GPU（如果使用GPU）
                    if self.device == 'gpu':
                        paddle.device.cuda.synchronize()

                    start_time = time.perf_counter()
                    prediction = self.model(input_tensor)

                    # 同步GPU确保计算完成
                    if self.device == 'gpu':
                        paddle.device.cuda.synchronize()

                    end_time = time.perf_counter()

                    inference_time = (end_time - start_time) * 1000  # 转换为毫秒
                    batch_times.append(inference_time)
                    all_times.append(inference_time)

                    if return_predictions:
                        batch_predictions.append(prediction)
                        all_predictions.append(prediction)

                    # 进度显示
                    if (i + 1) % (max(1, current_batch_size // 10)) == 0:
                        print(f"批次 {batch_idx + 1} 进度: {i + 1}/{current_batch_size}, 当前用时: {inference_time:.2f}ms")

            # 计算当前批次统计
            batch_mean = np.mean(batch_times)
            batch_std = np.std(batch_times)
            batch_min = np.min(batch_times)
            batch_max = np.max(batch_times)
            
            batch_stat = {
                'batch_id': batch_idx + 1,
                'batch_size': current_batch_size,
                'mean_ms': batch_mean,
                'std_ms': batch_std,
                'min_ms': batch_min,
                'max_ms': batch_max,
                'times': batch_times
            }
            batch_stats.append(batch_stat)
            
            print(f"批次 {batch_idx + 1} 完成 - 平均用时: {batch_mean:.2f}±{batch_std:.2f}ms")
            
            # 批次间等待（除了最后一个批次）
            if batch_idx < num_batches - 1:
                print(f"等待 {wait_between_batches} 秒后开始下一批次...")
                time.sleep(wait_between_batches)

        # 整体统计分析
        mean_time = np.mean(all_times)
        std_time = np.std(all_times)
        min_time = np.min(all_times)
        max_time = np.max(all_times)
        median_time = np.median(all_times)
        p95_time = np.percentile(all_times, 95)
        p99_time = np.percentile(all_times, 99)

        results = {
            'input_shape': list(input_tensor.shape),
            'target_size': target_size,
            'total_runs': len(all_times),
            'planned_runs': total_runs,
            'batch_size': batch_size,
            'num_batches': num_batches,
            'warmup_per_batch': warmup_per_batch,
            'wait_between_batches': wait_between_batches,
            'times_ms': all_times,
            'batch_statistics': batch_stats,
            'overall_statistics': {
                'mean_ms': mean_time,
                'std_ms': std_time,
                'min_ms': min_time,
                'max_ms': max_time,
                'median_ms': median_time,
                'p95_ms': p95_time,
                'p99_ms': p99_time,
                'fps': 1000 / mean_time,
                'throughput_imgs_per_sec': input_tensor.shape[0] * 1000 / mean_time
            }
        }

        if return_predictions:
            results['predictions'] = all_predictions

        # 打印结果
        self._print_batch_results(results)

        return results

    def _print_batch_results(self, results: Dict) -> None:
        """打印批次测试结果"""
        stats = results['overall_statistics']
        batch_stats = results['batch_statistics']

        print("\n" + "=" * 80)
        print("批次预测性能测试结果")
        print("=" * 80)
        print(f"输入尺寸: {results['input_shape']}")
        print(f"目标尺寸: {results['target_size']}")
        print(f"总测试次数: {results['total_runs']} (计划: {results['planned_runs']})")
        print(f"批次大小: {results['batch_size']}")
        print(f"批次数量: {results['num_batches']}")
        print(f"每批次预热: {results['warmup_per_batch']}")
        print(f"批次间等待: {results['wait_between_batches']}秒")
        print("-" * 80)
        
        # 整体统计
        print("整体统计:")
        print(f"平均用时: {stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms")
        print(f"中位数用时: {stats['median_ms']:.2f} ms")
        print(f"最小用时: {stats['min_ms']:.2f} ms")
        print(f"最大用时: {stats['max_ms']:.2f} ms")
        print(f"95%分位数: {stats['p95_ms']:.2f} ms")
        print(f"99%分位数: {stats['p99_ms']:.2f} ms")
        print(f"FPS: {stats['fps']:.2f}")
        print(f"吞吐量: {stats['throughput_imgs_per_sec']:.2f} images/sec")
        
        print("-" * 80)
        print("各批次统计:")
        for batch_stat in batch_stats:
            print(f"批次 {batch_stat['batch_id']}: {batch_stat['mean_ms']:.2f}±{batch_stat['std_ms']:.2f}ms "
                  f"(min: {batch_stat['min_ms']:.2f}, max: {batch_stat['max_ms']:.2f})")
        
        print("=" * 80)

    def predict_with_timing(self,
                            input_data,
                            target_size: Tuple[int, int],
                            num_runs: int = 50,
                            warmup_runs: int = 10,
                            return_predictions: bool = False) -> Dict:
        """
        对指定尺寸的数据进行预测，包含预热和多次测试取平均值

        Args:
            input_data: 输入数据，可以是图像路径、numpy数组或paddle张量
            target_size (Tuple[int, int]): 目标尺寸 (height, width)
            num_runs (int): 测试运行次数
            warmup_runs (int): 预热运行次数
            return_predictions (bool): 是否返回预测结果

        Returns:
            Dict: 包含预测结果和性能统计的字典
        """
        # 准备输入张量
        if isinstance(input_data, str):  # 图像路径
            input_tensor = self._preprocess_image(input_data, target_size)
        elif isinstance(input_data, np.ndarray):  # numpy数组
            input_tensor = self._preprocess_image(input_data, target_size)
        elif isinstance(input_data, paddle.Tensor):  # paddle张量
            input_tensor = input_data
        elif input_data is None:  # 生成虚拟数据
            input_tensor = self._generate_dummy_input(1, target_size[0], target_size[1])
        else:
            raise ValueError(f"不支持的输入数据类型: {type(input_data)}")

        print(f"输入尺寸: {list(input_tensor.shape)}")
        print(f"目标尺寸: {target_size}")

        # 预热
        self.warmup(input_tensor, warmup_runs)

        # 多次预测测试
        print(f"开始预测测试，运行 {num_runs} 次...")

        times = []
        predictions = []

        with paddle.no_grad():
            for i in range(num_runs):
                # 同步GPU（如果使用GPU）
                if self.device == 'gpu':
                    paddle.device.cuda.synchronize()

                start_time = time.perf_counter()
                prediction = self.model(input_tensor)

                # 同步GPU确保计算完成
                if self.device == 'gpu':
                    paddle.device.cuda.synchronize()

                end_time = time.perf_counter()

                inference_time = (end_time - start_time) * 1000  # 转换为毫秒
                times.append(inference_time)

                if return_predictions:
                    predictions.append(prediction)

                # 进度显示
                if (i + 1) % (max(1, num_runs // 10)) == 0:
                    print(f"进度: {i + 1}/{num_runs}, 当前用时: {inference_time:.2f}ms")

        # 统计分析
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        median_time = np.median(times)
        p95_time = np.percentile(times, 95)
        p99_time = np.percentile(times, 99)

        results = {
            'input_shape': list(input_tensor.shape),
            'target_size': target_size,
            'num_runs': num_runs,
            'warmup_runs': warmup_runs,
            'times_ms': times,
            'statistics': {
                'mean_ms': mean_time,
                'std_ms': std_time,
                'min_ms': min_time,
                'max_ms': max_time,
                'median_ms': median_time,
                'p95_ms': p95_time,
                'p99_ms': p99_time,
                'fps': 1000 / mean_time,
                'throughput_imgs_per_sec': input_tensor.shape[0] * 1000 / mean_time
            }
        }

        if return_predictions:
            results['predictions'] = predictions

        # 打印结果
        self._print_results(results)

        return results

    def predict_batch_sizes(self,
                            target_size: Tuple[int, int],
                            batch_sizes: List[int] = [1, 2, 4, 8],
                            num_runs: int = 30) -> Dict:
        """
        测试不同批次大小的预测性能

        Args:
            target_size (Tuple[int, int]): 目标尺寸 (height, width)
            batch_sizes (List[int]): 要测试的批次大小列表
            num_runs (int): 每个批次大小的运行次数

        Returns:
            Dict: 包含所有批次大小测试结果的字典
        """
        results = {}

        for batch_size in batch_sizes:
            print(f"\n=== 测试批次大小: {batch_size} ===")

            # 生成对应批次大小的输入
            input_tensor = self._generate_dummy_input(batch_size, target_size[0], target_size[1])

            # 预热
            self.warmup(input_tensor, warmup_runs=5)

            # 测试
            batch_results = self.predict_with_timing(
                input_tensor, target_size, num_runs, warmup_runs=0
            )

            results[f"batch_{batch_size}"] = batch_results

            # 清理显存
            del input_tensor
            if self.device == 'gpu':
                paddle.device.cuda.empty_cache()

        return results

    def _print_results(self, results: Dict) -> None:
        """打印预测结果"""
        stats = results['statistics']

        print("\n" + "=" * 60)
        print("预测性能测试结果")
        print("=" * 60)
        print(f"输入尺寸: {results['input_shape']}")
        print(f"目标尺寸: {results['target_size']}")
        print(f"测试次数: {results['num_runs']}")
        print(f"预热次数: {results['warmup_runs']}")
        print("=" * 60)

    def save_results(self, results: Dict, output_path: str) -> None:
        """保存结果到文件"""
        import json

        # 转换numpy类型为Python原生类型
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        results_json = convert_numpy(results)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)

        print(f"结果已保存到: {output_path}")


# 使用示例函数
def predict_model_performance(config_path: str,
                              model_path: str,
                              target_size: Tuple[int, int] = (512, 512),
                              input_data=None,
                              device: str = 'gpu',
                              num_runs: int = 50,
                              warmup_runs: int = 10) -> Dict:
    """
    便捷的模型性能预测函数

    Args:
        config_path (str): 配置文件路径
        model_path (str): 模型参数文件路径
        target_size (Tuple[int, int]): 目标尺寸 (height, width)
        input_data: 输入数据，None表示使用虚拟数据
        device (str): 设备类型
        num_runs (int): 测试运行次数
        warmup_runs (int): 预热运行次数

    Returns:
        Dict: 预测结果和性能统计
    """
    predictor = ModelPredictor(config_path, model_path, device)
    results = predictor.predict_with_timing(
        input_data, target_size, num_runs, warmup_runs
    )
    return results


if __name__ == '__main__':
    # 配置参数
    config_path = 'configs/quick_start/portrait_pp_humansegv2_lite.yml'
    model_path = 'output/portrait_pp_humansegv2_lite/best_model/model.pdparams'

    # 创建预测器
    predictor = ModelPredictor(config_path, model_path, 'auto')

    print("开始进行10个批次的性能测试...")
    print("=" * 80)
    
    # 直接进行10个批次的测试
    # 总共5000次测试，分为10个批次，每批次500次
    # 每批次预热50次，批次间等待10秒
    batch_results = predictor.predict_with_batch_timing(
        input_data=None,  # 使用虚拟数据
        target_size=(256, 144),
        total_runs=5000,
        batch_size=500,
        warmup_per_batch=50,
        wait_between_batches=10
    )

    print("\n所有批次处理完成！")
    print("正在生成整体性能报告...")
    
    # 保存详细结果到JSON文件
    predictor.save_results(batch_results, 'batch_prediction_results.json')
    
    print("\n整体性能报告已生成完成！")
    print(f"详细结果已保存到: batch_prediction_results.json")