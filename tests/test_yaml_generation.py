import os
import yaml
import sys
from pathlib import Path

# 添加当前目录到 sys.path
sys.path.append(os.getcwd())

from config import Config

def test_yaml_export():
    cfg = Config()
    test_export_dir = "temp_test_export"
    os.makedirs(test_export_dir, exist_ok=True)
    
    class_names = ["class1", "class2", "class3", "class4", "class5", "class6", "class7"]
    
    try:
        yaml_path = cfg.create_dataset_config(test_export_dir, class_names)
        print(f"Generated YAML path: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print("--- Generated File Content ---")
            print(content)
            print("------------------------------")
            
            # 重新加载以验证类型
            data = yaml.safe_load(content)
            print(f"Parsed Data: {data}")
            
            if not isinstance(data['nc'], int):
                print(f"FAILED: 'nc' should be int, but got {type(data['nc'])}")
                sys.exit(1)
            
            if data['nc'] != 7:
                print(f"FAILED: 'nc' should be 7, but got {data['nc']}")
                sys.exit(1)
                
            if not isinstance(data['names'], list):
                print(f"FAILED: 'names' should be list, but got {type(data['names'])}")
                sys.exit(1)
                
            print("SUCCESS: YAML generation verified!")
            
    finally:
        # 清理
        import shutil
        if os.path.exists(test_export_dir):
            shutil.rmtree(test_export_dir)

if __name__ == "__main__":
    test_yaml_export()
