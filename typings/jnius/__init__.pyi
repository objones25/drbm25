# Minimal stub: pyjnius 1.9.0 ships no stubs and re-exports these names from its
# compiled jnius.so via a star import, which pyright cannot see. Only the names
# drbm25 uses are declared.

class JavaException(Exception):
    classname: str

def detach() -> None: ...
