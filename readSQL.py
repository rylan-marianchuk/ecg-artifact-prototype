import sqlite3
from extract import *
from decodeLeads import getLeads

con = sqlite3.connect("/home/rylan/CLionProjects/preprocess/ecgArtifact.db")
cur = con.cursor()
print()

