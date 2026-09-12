def recommend(hardware, model):
    result = {'model': model['id'], 'status': 'unknown', 'profile': None,
              'reason': '', 'validated': model['validated'], 'defaults': model['defaults']}
    if not hardware['gpus']:
        result.update(status='not-recommended', reason='This adapter requires NVIDIA CUDA; no NVIDIA device was detected.')
        return result
    gpu = max(hardware['gpus'], key=lambda g: g['free_vram_gib'])
    result['gpu_index'] = gpu['index']
    available = gpu['free_vram_gib']
    minimum = model.get('min_vram_gib')
    recommended = model.get('recommended_vram_gib')
    if minimum is not None and available < minimum:
        result.update(status='probably-incompatible', reason='Available VRAM is below the documented minimum.')
    elif recommended is not None and available >= recommended:
        result.update(status='candidate', profile='gpu', reason='Meets documented VRAM recommendation; runtime checks still required.')
    else:
        result.update(profile='model-offload', reason='Unmeasured memory requirements: offload is a candidate, not a compatibility guarantee. RAM requirements are unknown.')
    if not hardware['torch'].get('cuda_available'):
        result['reason'] += ' Install a CUDA-enabled PyTorch build before execution.'
    return result
