select
    star.expnum,
    star.ccdnum,
    qa.psf_fwhm,
    e.band,
    star.ra,
    star.dec,
    star.u,
    star.v,
    star.x,
    star.y,
    star.flux,
    star.bdf_t,
    star.t_data,
    star.t_model,
    star.g1_data,
    star.g1_model,
    star.g2_data,
    star.g2_model,
    star.gi_color,
    star.iz_color,
    star.k_mag,
    star.g_mag,
    star.r_mag,
    star.i_mag,
    star.z_mag,
    star.flag_color,
    star.gaia_star
from 
    PROD.PIFF_HSM_STAR_QA star,
    PROD.PIFF_HSM_MODEL_QA model,
    QA_SUMMARY qa,
    exposure e,
    proctag t
where
    t.tag = 'Y6A2_PIFF_V3_TEST_01'
    and t.pfw_attempt_id=star.pfw_attempt_id
    and star.expnum = e.expnum
    and star.expnum = qa.expnum
    and e.program!='supernova'
    and star.model_filename = model.filename
    and model.flag = 0
    and model.nstar >= 30
    and star.is_reserve = 1
    and e.band != 'Y'
    and star.flag_model = 0
    and star.flag_truth = 0;
