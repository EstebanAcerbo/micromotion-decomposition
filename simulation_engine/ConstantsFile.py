from enum import IntEnum
class SystemConst(IntEnum):
    # A nivel pixelar
    EMPTY = 0
    MEMBRANE = 1
    INF_MEMBRANE = 2    # infected cell interior (alpha_inf, Cm_inf)
    MEM_FUS = 4         # fused cell interior (Cm_fus)
    TJ = 8
    TJ_INF_BORDER = 16  # TJ adjacent to >=1 infected cell (Rb_inf dominates)
    TJ_FUS = 32         # outer TJ of a fused island (scaled Rb)
    # A nivel Celular
    ALIVE         = 1
    DEAD          = 0
    FUSED         = 2
    INFECTED_CELL = 3   # infected cell (does not collide with FUSED=2)
    #Electrode
    ELECTRODE = 1
    INSULATOR = 0