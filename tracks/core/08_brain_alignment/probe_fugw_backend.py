"""Print FUGW's public API surface so the writeup can name the call site."""
import inspect
import importlib.metadata as md

print(f"fugw version: {md.version('fugw')}")
import fugw
print(f"top-level: {[x for x in dir(fugw) if not x.startswith('_')]}")

# The mappings module is the standard entry point in recent versions.
import fugw.mappings as fm
print(f"mappings: {[x for x in dir(fm) if not x.startswith('_')]}")

for cls_name in ("FUGW", "FUGWBarycenter", "FUGWSparse"):
    cls = getattr(fm, cls_name, None)
    if cls is None: continue
    print(f"\n=== {cls_name} ===")
    print(f"  __init__ sig: {inspect.signature(cls.__init__)}")
    if hasattr(cls, "fit"):
        print(f"  fit sig:      {inspect.signature(cls.fit)}")
    if hasattr(cls, "transform"):
        print(f"  transform sig: {inspect.signature(cls.transform)}")
