python ../../software/PLACER/PLACER.py -f inputs/super_af2_bu2.pdb -n 50 --odir outputs/ --prefix apo_ --ocsv outputs/super_af2_bu2_apo.csv --no-use_sm 
python ../../software/PLACER/PLACER.py -f inputs/super_af2_bu2.pdb -n 50 --odir outputs/ --prefix substrate_ --ocsv outputs/super_af2_bu2_substrate.csv 
python ../../software/PLACER/PLACER.py -f inputs/super_af2_bu2.pdb -n 50 --odir outputs/ --prefix tet1_ --ocsv outputs/super_af2_bu2_tet1.csv --mutate 128A:899 --no-use_sm 
python ../../software/PLACER/PLACER.py -f inputs/super_af2_bu2.pdb -n 50 --odir outputs/ --prefix aei_ --ocsv outputs/super_af2_bu2_aei.csv --mutate 128A:432 --no-use_sm 
python ../../software/PLACER/PLACER.py -f inputs/super_af2_bu2.pdb -n 50 --odir outputs/ --prefix tet2_ --ocsv outputs/super_af2_bu2_tet2.csv --mutate 128A:75I --no-use_sm 
