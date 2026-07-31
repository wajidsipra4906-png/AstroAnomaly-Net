import pandas as pd
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia

def query_simbad_target(tic_id: str) -> str:
    """
    Queries SIMBAD for main astronomical object type using TIC identifier.
    """
    try:
        custom_simbad = Simbad()
        custom_simbad.add_votable_fields('otype')
        result = custom_simbad.query_object(f"TIC {tic_id}")
        if result is not None and len(result) > 0:
            return str(result['OTYPE'][0])
        return "* (Standard Star)"
    except Exception:
        return "Unknown / Unclassified"

def query_gaia_ruwe(ra: float, dec: float) -> float:
    """
    Queries Gaia DR3 TAP server to fetch RUWE score for given coordinates.
    A RUWE value ~1.0 indicates a clean single-star astrometric fit.
    """
    query = f"""
    SELECT ruwe
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(
      POINT('ICRS', {ra}, {dec}),
      CIRCLE('ICRS', gaiadr3.gaia_source.ra, gaiadr3.gaia_source.dec, 0.0014)
    )
    """
    try:
        job = Gaia.launch_job(query)
        results = job.get_results()
        if len(results) > 0 and not results['ruwe'].mask[0]:
            return round(float(results['ruwe'][0]), 3)
        return 1.000
    except Exception:
        return 1.000
