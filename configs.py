configs = [
    {'name':'000','prefix':'n', 'epochs':32, 'learn_rate':0.0005, 'k':5, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'001','prefix':'n_eh', 'epochs':32, 'learn_rate':0.0005, 'k':5, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'002','prefix':'n_ec', 'epochs':32, 'learn_rate':0.0005, 'k':5, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'003','prefix':'n_ea', 'epochs':32, 'learn_rate':0.0005, 'k':5, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'004','prefix':'n_fr', 'epochs':32, 'learn_rate':0.0005, 'k':5, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'005','prefix':'n_eh_fr', 'epochs':32, 'learn_rate':0.0005, 'k':5, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'006','prefix':'n_ec_fr', 'epochs':32, 'learn_rate':0.0005, 'k':5, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'007','prefix':'n_ea_fr', 'epochs':32, 'learn_rate':0.0005, 'k':5, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },

    #learn rate: best: 0.0007
    {'name':'008','prefix':'n_ea_fr', 'epochs':32, 'learn_rate':0.0001, 'k':5, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'009','prefix':'n_ea_fr', 'epochs':32, 'learn_rate':0.0002, 'k':5, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'010','prefix':'n_ea_fr', 'epochs':32, 'learn_rate':0.0003, 'k':5, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'011','prefix':'n_ea_fr', 'epochs':32, 'learn_rate':0.0004, 'k':5, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'012','prefix':'n_ea_fr', 'epochs':32, 'learn_rate':0.0005, 'k':5, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'013','prefix':'n_ea_fr', 'epochs':32, 'learn_rate':0.0006, 'k':5, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'014','prefix':'n_ea_fr', 'epochs':32, 'learn_rate':0.0007, 'k':5, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'015','prefix':'n_ea_fr', 'epochs':32, 'learn_rate':0.0008, 'k':5, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'016','prefix':'n_ea_fr', 'epochs':32, 'learn_rate':0.0009, 'k':5, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'017','prefix':'n_ea_fr', 'epochs':32, 'learn_rate':0.001, 'k':5, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },

    # epochs: best: k: 3
    {'name':'018','prefix':'n_ea_fr', 'epochs':32, 'learn_rate':0.0005, 'k':2, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'019','prefix':'n_ea_fr', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'020','prefix':'n_ea_fr', 'epochs':32, 'learn_rate':0.0005, 'k':4, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'021','prefix':'n_ea_fr', 'epochs':32, 'learn_rate':0.0005, 'k':5, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'022','prefix':'n_ea_fr', 'epochs':32, 'learn_rate':0.0005, 'k':6, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'023','prefix':'n_ea_fr', 'epochs':32, 'learn_rate':0.0005, 'k':7, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },

    # fcr1 size: best: 4096
    {'name':'024','prefix':'n_ea_fr', 'epochs':32, 'learn_rate':0.0005, 'k':5, 'fcr1_size':1024*16, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'025','prefix':'n_ea_fr', 'epochs':32, 'learn_rate':0.0005, 'k':5, 'fcr1_size':1024*8, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'026','prefix':'n_ea_fr', 'epochs':32, 'learn_rate':0.0005, 'k':5, 'fcr1_size':1024*4, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'027','prefix':'n_ea_fr', 'epochs':32, 'learn_rate':0.0005, 'k':5, 'fcr1_size':1024*2, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'028','prefix':'n_ea_fr', 'epochs':32, 'learn_rate':0.0005, 'k':5, 'fcr1_size':1024*1, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },

    # fcr2 size: best: 1024
    {'name':'029','prefix':'n_ea_fr', 'epochs':32, 'learn_rate':0.0005, 'k':7, 'fcr1_size':2048, 'fcr2_size':128*8, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'030','prefix':'n_ea_fr', 'epochs':32, 'learn_rate':0.0005, 'k':7, 'fcr1_size':2048, 'fcr2_size':128*4, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'031','prefix':'n_ea_fr', 'epochs':32, 'learn_rate':0.0005, 'k':7, 'fcr1_size':2048, 'fcr2_size':128*2, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'032','prefix':'n_ea_fr', 'epochs':32, 'learn_rate':0.0005, 'k':7, 'fcr1_size':2048, 'fcr2_size':128*1, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },

    # cr size: best: 64 128 256
    {'name':'033','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':5, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':2, 'cr2_d':4, 'cr3_d':8, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'034','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':5, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':4, 'cr2_d':8, 'cr3_d':16, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'035','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':5, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':8, 'cr2_d':16, 'cr3_d':32, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'036','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':5, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':16, 'cr2_d':32, 'cr3_d':64, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'037','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':5, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'038','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':5, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':64, 'cr2_d':128, 'cr3_d':256, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },

    #cr1 dropout: best: 0.6
    {'name':'039','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.4, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'040','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.5, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'041','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.6, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'042','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.7, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'043','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.8, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'044','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.9, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },

    #cr2 dropout: best: 0.5
    {'name':'045','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':0.4, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'046','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':0.5, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'047','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':0.6, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'048','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':0.7, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'049','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':0.8, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'050','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':0.9, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },

    # cr3 dropout: best: 0.6
    {'name':'051','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':0.4, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'052','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':0.5, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'053','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':0.6, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'054','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':0.7, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'055','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':0.8, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'056','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':0.9, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },

    #fcr1 dropout: best: 0.6
    {'name':'057','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':0.4, 'fcr2_drop':1.0 },
    {'name':'058','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':0.5, 'fcr2_drop':1.0 },
    {'name':'059','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':0.6, 'fcr2_drop':1.0 },
    {'name':'060','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':0.7, 'fcr2_drop':1.0 },
    {'name':'061','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':0.8, 'fcr2_drop':1.0 },
    {'name':'062','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':0.9, 'fcr2_drop':1.0 },

    # fcr2 dropout: not much diff, best 0.7
    {'name':'063','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':0.4 },
    {'name':'064','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':0.5 },
    {'name':'065','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':0.6 },
    {'name':'066','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':0.7 },
    {'name':'067','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':0.8 },
    {'name':'068','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':1.0, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':0.9 },


    {'name':'068','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.5, 'cr2_drop':0.6, 'cr3_drop':0.7, 'fcr1_drop':0.6, 'fcr2_drop':0.7 },
    {'name':'069','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.5, 'cr2_drop':0.6, 'cr3_drop':0.7, 'fcr1_drop':0.7, 'fcr2_drop':0.8 },
    {'name':'070','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.5, 'cr2_drop':0.6, 'cr3_drop':0.7, 'fcr1_drop':0.8, 'fcr2_drop':0.9 },
    {'name':'071','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.5, 'cr2_drop':0.6, 'cr3_drop':0.7, 'fcr1_drop':0.9, 'fcr2_drop':1.0 },


    {'name':'072','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.5, 'cr2_drop':0.6, 'cr3_drop':0.7, 'fcr1_drop':0.7, 'fcr2_drop':0.7 },
    {'name':'073','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.5, 'cr2_drop':0.6, 'cr3_drop':0.7, 'fcr1_drop':0.8, 'fcr2_drop':0.8 },
    {'name':'074','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.5, 'cr2_drop':0.6, 'cr3_drop':0.7, 'fcr1_drop':0.9, 'fcr2_drop':0.9 },
    {'name':'075','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.5, 'cr2_drop':0.6, 'cr3_drop':0.7, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },

    {'name':'076','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.4, 'cr2_drop':0.6, 'cr3_drop':0.8, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'077','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.5, 'cr2_drop':0.7, 'cr3_drop':0.9, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'078','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.6, 'cr2_drop':0.8, 'cr3_drop':1.0, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },

    {'name':'079','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':1.0, 'cr2_drop':0.8, 'cr3_drop':0.6, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'080','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.9, 'cr2_drop':0.7, 'cr3_drop':0.5, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'081','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.8, 'cr2_drop':0.6, 'cr3_drop':0.4, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },

    {'name':'082','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.3, 'cr2_drop':0.4, 'cr3_drop':0.5, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'083','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.4, 'cr2_drop':0.5, 'cr3_drop':0.6, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'084','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.5, 'cr2_drop':0.6, 'cr3_drop':0.7, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'085','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.6, 'cr2_drop':0.7, 'cr3_drop':0.8, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'086','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.7, 'cr2_drop':0.8, 'cr3_drop':0.9, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },

    {'name':'087','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.5, 'cr2_drop':0.4, 'cr3_drop':0.3, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'088','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.6, 'cr2_drop':0.5, 'cr3_drop':0.4, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'089','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.7, 'cr2_drop':0.5, 'cr3_drop':0.5, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'090','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.8, 'cr2_drop':0.6, 'cr3_drop':0.6, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'091','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.9, 'cr2_drop':0.7, 'cr3_drop':0.7, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },

    {'name':'092','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.3, 'cr2_drop':0.3, 'cr3_drop':0.3, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'093','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.4, 'cr2_drop':0.4, 'cr3_drop':0.4, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'094','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.5, 'cr2_drop':0.5, 'cr3_drop':0.5, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'095','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.6, 'cr2_drop':0.6, 'cr3_drop':0.6, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'096','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.7, 'cr2_drop':0.7, 'cr3_drop':0.7, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'097','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.8, 'cr2_drop':0.8, 'cr3_drop':0.8, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },
    {'name':'098','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.9, 'cr2_drop':0.9, 'cr3_drop':0.9, 'fcr1_drop':1.0, 'fcr2_drop':1.0 },

    # filters best: noblur nonoise
    {'name':'099','prefix':'n_ea_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.9, 'cr2_drop':0.8, 'cr3_drop':0.7, 'fcr1_drop':0.5, 'fcr2_drop':1.0 },
    {'name':'100','prefix':'n_ea_f_noblur', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.9, 'cr2_drop':0.8, 'cr3_drop':0.7, 'fcr1_drop':0.5, 'fcr2_drop':1.0 },
    {'name':'101','prefix':'n_ea_f_noblur_nonoise', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.9, 'cr2_drop':0.8, 'cr3_drop':0.7, 'fcr1_drop':0.5, 'fcr2_drop':1.0 },
    {'name':'102','prefix':'n_ea_f_noblur_large_noisefirst', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.9, 'cr2_drop':0.8, 'cr3_drop':0.7, 'fcr1_drop':0.5, 'fcr2_drop':1.0 },

    {'name':'103','prefix':'n_ea_fr_f', 'epochs':32, 'learn_rate':0.0005, 'k':3, 'fcr1_size':2048, 'fcr2_size':512, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.9, 'cr2_drop':0.8, 'cr3_drop':0.7, 'fcr1_drop':0.5, 'fcr2_drop':1.0 },


    {'name':'104','prefix':'n_ea_fr_f', 'epochs':32, 'learn_rate':0.0007, 'k':3, 'fcr1_size':4096, 'fcr2_size':1024, 'cr1_d':64, 'cr2_d':128, 'cr3_d':256, 'cr1_drop':0.6, 'cr2_drop':0.5, 'cr3_drop':0.6, 'fcr1_drop':0.6, 'fcr2_drop':1.0 },

    # l2 [fcr1_weights, fcr2_weights, fc3_weights] 0.0001*
    {'name':'105','prefix':'n_ea_fr_f', 'epochs':64, 'learn_rate':0.0007, 'k':3, 'fcr1_size':4096, 'fcr2_size':1024, 'cr1_d':32, 'cr2_d':64, 'cr3_d':128, 'cr1_drop':0.6, 'cr2_drop':0.5, 'cr3_drop':0.6, 'fcr1_drop':0.6, 'fcr2_drop':0.7 },

    # softmax
    {'106':'_dev_','prefix':'n_ea_fr_f', 'epochs':128, 'learn_rate':0.0007, 'k':3, 'fcr1_size':4096, 'fcr2_size':1024, 'cr1_d':64, 'cr2_d':128, 'cr3_d':256, 'cr1_drop':0.6, 'cr2_drop':0.5, 'cr3_drop':0.6, 'fcr1_drop':0.6, 'fcr2_drop':0.7 },

    #sigmoid
    {'name':'_dev_','prefix':'n_ea_fr_f', 'epochs':128, 'learn_rate':0.0007, 'k':3, 'fcr1_size':4096, 'fcr2_size':1024, 'cr1_d':64, 'cr2_d':128, 'cr3_d':256, 'cr1_drop':0.6, 'cr2_drop':0.5, 'cr3_drop':0.6, 'fcr1_drop':0.6, 'fcr2_drop':0.7 }

]