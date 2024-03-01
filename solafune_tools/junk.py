import pystac
import statistics

catalog = pystac.Catalog.from_file("data/stac/catalog.json")
items = list(catalog.get_items(recursive=True))
epsg_list = []
for item in items:
    epsg_list.append(item.to_dict()['properties']['proj:epsg'])
print(statistics.mode(epsg_list))