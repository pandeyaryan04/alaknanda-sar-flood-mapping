"""
merge_tiles.py  --  merge adjacent single-band rasters into one GeoTIFF.

Written for the two CartoDEM 30 m tiles, E078 and E079, which have to become
one raster before 07_dem_comparison.py can use CartoDEM as a candidate.

  python merge_tiles.py --out DEM_CARTODEM_30m.tif tileA.tif tileB.tif
"""
import argparse, os, sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nodata", type=float, default=None)
    ap.add_argument("tiles", nargs="+")
    a = ap.parse_args()

    import rasterio
    from rasterio.merge import merge

    missing = [t for t in a.tiles if not os.path.exists(t)]
    if missing:
        print("missing:"); [print("  " + m) for m in missing]
        return 1

    srcs = [rasterio.open(t) for t in a.tiles]
    crss = {str(s.crs) for s in srcs}
    if len(crss) > 1:
        print("tiles are in different CRSs, refusing to merge: %s" % crss)
        return 1
    nod = a.nodata if a.nodata is not None else srcs[0].nodata
    mosaic, tr = merge(srcs, nodata=nod)
    prof = dict(srcs[0].profile, height=mosaic.shape[1], width=mosaic.shape[2],
                transform=tr, count=1, compress="lzw", nodata=nod)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with rasterio.open(a.out, "w", **prof) as dst:
        dst.write(mosaic[0], 1)
    for s in srcs:
        s.close()
    print("wrote %s  %d x %d  nodata %s"
          % (a.out, mosaic.shape[1], mosaic.shape[2], nod))
    return 0


if __name__ == "__main__":
    sys.exit(main())
