import json
import tempfile
from pathlib import Path
from config import Config
import importlib.util
import sys
from pathlib import Path as _Path


# Load core.project_manager by decoding source as UTF-8 (ignore errors) to avoid
# potential encoding issues when pytest re-imports modules.
def _load_project_manager():
    spec = importlib.util.spec_from_file_location("core.project_manager", str(_Path(__file__).parents[1] / 'core' / 'project_manager.py'))
    module = importlib.util.module_from_spec(spec)
    with open(spec.origin, 'rb') as f:
        src = f.read()
    src = src.decode('utf-8', errors='ignore')
    exec(compile(src, spec.origin, 'exec'), module.__dict__)
    sys.modules[spec.name] = module
    return module.ProjectManager


ProjectManager = _load_project_manager()
from PyQt5.QtGui import QColor


def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def test_get_classes_from_txt_simple(tmp_path):
    cfg = Config(config_file=str(tmp_path / 'config.json'))
    cfg.app_config.output_dir = str(tmp_path / 'workspace')
    cfg.ensure_dirs()

    classes_txt = Path(cfg.get_workspace_path()) / 'classes.txt'
    write_file(classes_txt, "cat\ndog\n")

    pm = ProjectManager(cfg)
    classes = pm.get_classes()
    assert len(classes) == 2
    assert classes[0].name == 'cat'
    assert classes[1].name == 'dog'
    assert isinstance(classes[0].color, QColor)


def test_get_classes_from_txt_with_colors(tmp_path):
    cfg = Config(config_file=str(tmp_path / 'config.json'))
    cfg.app_config.output_dir = str(tmp_path / 'workspace')
    cfg.ensure_dirs()

    classes_txt = Path(cfg.get_workspace_path()) / 'classes.txt'
    write_file(classes_txt, "cat,#112233\ndog,#445566\n")

    pm = ProjectManager(cfg)
    classes = pm.get_classes()
    assert len(classes) == 2
    assert classes[0].name == 'cat'
    assert classes[0].color.name().lower() == '#112233'
    assert classes[1].color.name().lower() == '#445566'


def test_get_classes_from_json_list_strings(tmp_path):
    cfg = Config(config_file=str(tmp_path / 'config.json'))
    cfg.app_config.output_dir = str(tmp_path / 'workspace')
    cfg.ensure_dirs()

    classes_json = Path(cfg.get_workspace_path()) / 'classes.json'
    write_file(classes_json, json.dumps(["alpha", "beta"]))

    pm = ProjectManager(cfg)
    classes = pm.get_classes()
    assert len(classes) == 2
    assert classes[0].name == 'alpha'
    assert classes[1].name == 'beta'


def test_get_classes_from_json_list_objects(tmp_path):
    cfg = Config(config_file=str(tmp_path / 'config.json'))
    cfg.app_config.output_dir = str(tmp_path / 'workspace')
    cfg.ensure_dirs()

    data = [
        {"id": 0, "name": "one", "color": "#abcdef"},
        {"id": 1, "name": "two", "color": "#123456"}
    ]
    classes_json = Path(cfg.get_workspace_path()) / 'classes.json'
    write_file(classes_json, json.dumps(data))

    pm = ProjectManager(cfg)
    classes = pm.get_classes()
    assert len(classes) == 2
    assert classes[0].name == 'one'
    assert classes[0].color.name().lower() == '#abcdef'
    assert classes[1].color.name().lower() == '#123456'
