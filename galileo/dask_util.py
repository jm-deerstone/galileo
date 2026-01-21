# 192.168.0.255
from dask.distributed import Client, get_client


def try_get_dask_client():

    try:

        try:
            return get_client()
        except ValueError:
            # Try to connect to a default scheduler on the local network (optional)
            # Replace the address if your scheduler isn't local
            try:
                return Client('tcp://192.168.0.37:8786')
            except Exception:
                return None
    except ImportError:
        return None

c = try_get_dask_client()
print("Dask client:", c)

# dask scheduler
# dask worker tcp://192.168.0.37:8786 --nworkers 12 --nthreads 1
