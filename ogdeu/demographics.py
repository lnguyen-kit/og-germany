# ogdeu/demography.py
from ogcore import demographics


# wrapper
DEU_CODE = "276"

def get_demog_S(p, download_path=None, graph=False):
    return demographics.get_pop_objs(
        p.E, p.S, p.T, 0, 99,
        country_id=DEU_CODE,
        initial_data_year=p.start_year - 1,
        final_data_year=p.start_year + 1,
        GraphDiag=graph,
        download_path=download_path,
    )

def get_demog_80(p, download_path=None, graph=False):
    return demographics.get_pop_objs(
        20, 80, p.T, 0, 99,
        country_id=DEU_CODE,
        initial_data_year=p.start_year - 1,
        final_data_year=p.start_year + 1,
        GraphDiag=graph,
        download_path=download_path,
    )
