################################
# PATH dictionnary
################################


# on PC
#---------------
PATH_DICT = {

    '/home/antlafe':{
    
        # Repertory where output pikle files of aligned data will be stored
        'PATH_OUT': '/home/antlafe/Documents/work/data/CRYO2ICE/Cryo2Ice/',
        # Repertory where all tracks from mission will be found
        'PATH_DATA': '/home/antlafe/Documents/work/data/',
        # Repertory where only collocated tracks will be found
        'PATH_COLLOC': '/home/antlafe/Documents/work/data/CRYO2ICE/',
        # Repertory to store gridded data
        'PATH_GRID': '/home/antlafe/Documents/work/grid_data/',
        # Repertory to store output figures
        'PATH_FIG': '/home/antlafe/Documents/work/figures/cryo2ice/',
        # repertory to store spreadsheet
        'PATH_SPREADSHEET': '/home/antlafe/Documents/work/projet_cryo2ice/data/',
        },


    '/home/il/laforga':{

        'PATH_OUT': '/work/ALT/odatis/seaice/users/laforga/data/CRYO2ICE/Cryo2Ice/',
        'PATH_DATA': '/work/ALT/odatis/seaice/users/laforga/data/',
        'PATH_COLLOC': '/work/ALT/odatis/seaice/users/laforga/data/CRYO2ICE/',
        'PATH_GRID': '/work/ALT/odatis/seaice/users/laforga/grid_data/',
        'PATH_FIG': '/work/ALT/odatis/seaice/users/laforga/figures/',
        'PATH_SPREADSHEET': '/work/ALT/odatis/seaice/users/laforga/projet_cryo2ice/',

        },
}




# list of days for which CS2/IS2 are collocated with one day apart
MIDNIGHT_DATES = {
    'ATL07 ': ['20201005','20201018','20201022','20201108','20201125','20201129','20201216','20210102','20210119','20210123','20210209','20210226','20210315','20210319','20210401','20210422'],
    'ATL10': ['20201005','20201018','20201022','20201108','20201125','20201129','20201216','20210102','20210119','20210123','20210209','20210226','20210315','20210319','20210401','20210422'],
    'ATL12': ['20201014','20201031'], #not up to date
    }

