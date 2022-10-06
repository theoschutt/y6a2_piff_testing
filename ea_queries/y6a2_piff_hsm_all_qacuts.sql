select qa.*, m.band 
from GRUENDL.PIFF_HSM_STAR_QA qa, PIFF_HSM_MODEL_QA qam, miscfile m
where m.filetype='piff_model'
    and m.filename=qam.filename
    and qa.model_filename = qam.filename
    and qam.flag = 0
    and qam.nstar >= 30
    and qa.reserve = 1
    and m.band != 'Y';