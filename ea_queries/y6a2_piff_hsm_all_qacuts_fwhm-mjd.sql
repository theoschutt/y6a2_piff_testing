select
    star.expnum,
    star.ccdnum,
    m.band,
    exp.mjd_obs,
    qa.psf_fwhm,
    star.ra,
    star.dec,
    star.u,
    star.v,
    star.x,
    star.y,
    star.flux,
    star.t_data,
    star.t_model,
    star.g1_data,
    star.g1_model,
    star.g2_data,
    star.g2_model,
    pqa.gi_color,
    pqa.iz_color
from 
    GRUENDL.PIFF_HSM_STAR_QA star,
    EXPOSURE exp,
    PIFF_HSM_MODEL_QA model,
    PIFF_STAR_QA pqa,
    QA_SUMMARY qa,
    miscfile m
where
    m.filetype='piff_model'
    and m.filename=model.filename
    and star.model_filename = model.filename
    and star.model_filename = pqa.filename
    and star.expnum = qa.expnum
    and star.expnum = exp.expnum
    and abs(star.ra - pqa.ra) < 1.E-6
    and abs(star.dec - pqa.dec) < 1.E-6
    and model.flag = 0
    and model.nstar >= 30
    and star.reserve = 1
    and m.band != 'Y';