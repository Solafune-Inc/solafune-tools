import geopandas as gpd
import shapely

gdf = gpd.read_parquet("tests/data-test/parquet/sample.parquet")
geom1 = gdf.geometry.iloc[0]
geom2 = gdf.geometry.iloc[2]

print(shapely.equals(geom1.union(geom2).union(geom1), geom2))