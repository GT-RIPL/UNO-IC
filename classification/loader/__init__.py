from loader.LT_loader import LT_Loader


def get_loader(name,**kargs):
    return {
        "place365": LT_Loader,
    }[name]
