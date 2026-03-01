"""Debug script to find correct Endee SDK params and test full create+upsert flow."""
import sys, inspect, json
sys.path.insert(0, ".")
from endee import Endee, Precision

c = Endee()
c.set_base_url("http://localhost:8080/api/v1")

# Write full signatures to file
out = {}
out["create_index_sig"] = str(inspect.signature(c.create_index))
out["get_index_sig"] = str(inspect.signature(c.get_index))

# Try create_index step by step
print("create_index signature:", out["create_index_sig"])
print("get_index signature:", out["get_index_sig"])

# Try calling with positional args based on signature order
# (name, dimension, space_type, precision, m, ef_construction, sparse_dim)
try:
    r = c.create_index("test_debug", 4, "cosine", Precision.INT8)
    print("Created with min params:", r)
except TypeError as e:
    print("TypeError min params:", e)
except Exception as e:
    print("API Error min params:", e)

# Now check upsert signature
try:
    idx = c.get_index("test_debug")
    print("Got index:", idx)
    src = inspect.getsource(idx.upsert)
    print("\nupsert source:", src[:1000])
except Exception as e:
    print("get_index error:", e)
