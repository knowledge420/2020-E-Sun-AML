class CaseInsensitiveDict(dict):
    def get(self, key, default=None):
        return super().get(key, default)

    def __getitem__(self, key):
        return super().__getitem__(key.lower())

    def __setitem__(self, key, value):
        super().__setitem__(key.lower(), value)

    def __contains__(self, key):
        return super().__contains__(key.lower())
