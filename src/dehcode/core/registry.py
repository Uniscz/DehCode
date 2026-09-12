import json
from importlib.resources import files


def models():
    result = []
    seen = set()
    for path in sorted(files('dehcode').joinpath('registry').iterdir(), key=lambda p: p.name):
        if path.name.endswith('.json'):
            model = json.loads(path.read_text(encoding='utf-8'))
            required = {'id','name','repository','adapter','backend','defaults','precisions','profiles','validated'}
            if not required <= model.keys() or model['id'] in seen:
                raise ValueError(f'Invalid or duplicate model registry: {path.name}')
            seen.add(model['id'])
            result.append(model)
    return result


def get_model(model_id):
    for model in models():
        if model['id'] == model_id:
            return model
    raise ValueError(f'Unknown model: {model_id}. Run dehcode models.')
