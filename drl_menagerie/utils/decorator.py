def api(func):
    func.is_api = True
    return func
