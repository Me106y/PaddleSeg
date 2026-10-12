import argparse
import paddle
import paddleseg
import importlib
import yaml
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.analyze_model import _dynamic_flops

def parse_args():
    parser = argparse.ArgumentParser(description='PaddleSeg Model Info')
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--model_path', default=None, help='模型权重路径，可选')
    parser.add_argument('--input_shape', type=int, nargs=3, default=[3, 256, 144], help='输入 shape，默认[3,512,512]')
    return parser.parse_args()

def parse_model(cfg):
    model_cfg = cfg['model']
    model_type = model_cfg.pop('type')
    # 自动补充 num_classes
    if 'num_classes' not in model_cfg:
        if 'train_dataset' in cfg and 'num_classes' in cfg['train_dataset']:
            model_cfg['num_classes'] = cfg['train_dataset']['num_classes']
        elif 'val_dataset' in cfg and 'num_classes' in cfg['val_dataset']:
            model_cfg['num_classes'] = cfg['val_dataset']['num_classes']
    # 递归实例化 backbone
    if 'backbone' in model_cfg and isinstance(model_cfg['backbone'], dict):
        backbone_cfg = model_cfg['backbone']
        backbone_type = backbone_cfg.pop('type')
        backbone_module = importlib.import_module('paddleseg.models.backbones')
        backbone_class = getattr(backbone_module, backbone_type)
        model_cfg['backbone'] = backbone_class(**backbone_cfg)
    model_module = importlib.import_module('paddleseg.models')
    model_class = getattr(model_module, model_type)
    return model_class(**model_cfg)

def load_model_from_config(config_path, model_path=None):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    model = parse_model(cfg)
    if model_path:
        state_dict = paddle.load(model_path)
        model.set_state_dict(state_dict)
    model.eval()
    return model

def main():
    args = parse_args()
    model = load_model_from_config(args.config, args.model_path)
    input_shape = [1] + args.input_shape
    print(f"模型结构: {model.__class__.__name__}")
    print(f"输入 shape: {input_shape}")
    dummy_input = paddle.randn(input_shape, dtype='float32')
    _dynamic_flops(model, dummy_input)

if __name__ == '__main__':
    main()