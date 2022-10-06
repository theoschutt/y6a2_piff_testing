select
    star.expnum,
    exp.mjd_obs,
    star.ra,
    star.dec
from 
    GRUENDL.PIFF_HSM_STAR_QA star,
    EXPOSURE exp,
    PIFF_HSM_MODEL_QA model,
    PIFF_STAR_QA pqa,
    miscfile m
where
    m.filetype='piff_model'
    and m.filename=model.filename
    and star.model_filename = model.filename
    and star.model_filename = pqa.filename
    and exp.expnum = star.expnum
    and abs(star.ra - pqa.ra) < 1.E-6
    and abs(star.dec - pqa.dec) < 1.E-6
    and model.flag = 0
    and model.nstar >= 30
    and star.reserve = 1
    and m.band != 'Y';