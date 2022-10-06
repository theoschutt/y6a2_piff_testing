select
    distinct(model.expnum),
    e.band,
    e.program,
    e.mjd_obs,
    e.telra,
    e.teldec,
    qa.psf_fwhm,
    qa.flag
from PROD.PIFF_HSM_MODEL_QA model
    join proctag t on model.pfw_attempt_id=t.pfw_attempt_id
    join exposure e on model.expnum=e.expnum
    join QA_SUMMARY qa on model.expnum=qa.expnum
where
    t.tag = 'Y6A2_PIFF_V3'
    and model.flag = 0
    and model.nstar >= 30;
