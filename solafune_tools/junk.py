from dask.distributed import LocalCluster, Client

if __name__=="__main__":
    cluster = LocalCluster()
    client = Client(cluster)
    # address = client.scheduler.address
    print(client)
    client.close()
    cluster.close()