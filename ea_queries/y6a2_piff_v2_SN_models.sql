select
    model.expnum,
    model.ccdnum,
    e.band,
    fai.path,
    fai.filename
from PROD.PIFF_HSM_MODEL_QA_DEPRECATED model
    join proctag t on model.pfw_attempt_id=t.pfw_attempt_id
    join file_archive_info fai on model.filename=fai.filename
    join exposure e on model.expnum=e.expnum
where
    t.tag = 'Y6A2_PIFF_V2'
    and e.program = 'supernova'
    and model.flag = 0;