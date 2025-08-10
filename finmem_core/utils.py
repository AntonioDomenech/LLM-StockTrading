import pathlib, datetime as dt
def stamp_dir(root: str) -> str:
    p = pathlib.Path(root)/dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    p.mkdir(parents=True, exist_ok=True)
    return str(p)
def ensure_dir(path: str):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
