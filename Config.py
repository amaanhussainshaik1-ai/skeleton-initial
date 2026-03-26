# This file contains some variable names you need to use in overall project. 
#For example, this will contain the name of dataframe columns we will working on each file
class Config:
    # Input Columns
    TICKET_SUMMARY = 'Ticket Summary'
    INTERACTION_CONTENT = 'Interaction content'

    # Raw label column names (as they appear in the CSV)
    TYPE1_COL = 'Type 1'
    TYPE2_COL = 'Type 2'
    TYPE3_COL = 'Type 3'
    TYPE4_COL = 'Type 4'

    # Type Columns to test
    TYPE_COLS = ['y2', 'y3', 'y4']
    CLASS_COL = 'y2'
    GROUPED = 'y1'

    # Chained label columns created for Design Choice 1
    CHAIN_Y2         = 'chain_y2'           # just Type2
    CHAIN_Y2_Y3      = 'chain_y2_y3'        # Type2 + Type3
    CHAIN_Y2_Y3_Y4   = 'chain_y2_y3_y4'    # Type2 + Type3 + Type4

    # Minimum number of samples a class must have to be kept
    MIN_CLASS_SAMPLES = 3
 
    # Train / test split ratio
    TEST_SIZE   = 0.2
    RANDOM_SEED = 0