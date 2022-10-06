select
    star.expnum,
    star.ccdnum,
    qa.psf_fwhm,
    m.band,
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
    pqa.star_flag,
    pqa.gi_color,
    pqa.iz_color
from
    y6a2_mdet_run1_piffmods_in_coadds pmods,
    GRUENDL.PIFF_HSM_STAR_QA star,
    PIFF_HSM_MODEL_QA model,
    PIFF_STAR_QA pqa,
    QA_SUMMARY qa,
    miscfile m,
    proctag t
where
    t.tag = 'Y6A2_PIFF_V2'
    and t.pfw_attempt_id=m.pfw_attempt_id
    and m.filetype='piff_model'
    and model.filename = pmods.filename
    and m.filename=model.filename
    and star.model_filename = model.filename
    and star.model_filename = pqa.filename
    and star.expnum = qa.expnum
    and abs(star.ra - pqa.ra) < 1.E-6
    and abs(star.dec - pqa.dec) < 1.E-6
    and model.flag = 0
    and model.nstar >= 30
    and star.reserve = 1
    and m.band != 'Y';
    and rownum<11;