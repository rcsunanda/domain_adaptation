"""
Tests utility functions
"""

# from domain_adaptation import utility
import domain_adaptation.utility as utility



###################################################################################################
"""
Write two lists to a CSV file and see if they are properly appended 
"""

def test_dump_data_to_csv():
    list1 = [11, 12, 13, 14, 15]
    utility.dump_data_to_csv("List1", list1)

    list2 = [500, 600, 700, 800, 900]
    utility.dump_data_to_csv("List2", list2)

    print("test_dump_data_to_csv; complete")



###################################################################################################

# Call test functions

test_dump_data_to_csv()