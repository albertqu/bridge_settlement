class BridgeNameConverter:
    regex = '[\w-]+'

    def to_python(self, value):
        return value.replace("-", " ").title()
    
    def to_url(self, value):
        return str(value).replace(" ", "-").lower()