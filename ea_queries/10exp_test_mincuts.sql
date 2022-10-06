select qa.*
from PROD.PIFF_HSM_STAR_QA qa, PROD.PIFF_HSM_MODEL_QA qam
where qa.model_filename = qam.filename
    and qam.flag = 0
    and qam.nstar >= 30;
