workclass = ['Private', 'Self-emp-not-inc', 'Self-emp-inc',
             'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
education = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc',
             '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']
maritalStatus = ['Married-civ-spouse', 'Divorced', 'Never-married',
                 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
occupation = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
              'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
relationship = ['Wife', 'Own-child', 'Husband',
                'Not-in-family', 'Other-relative', 'Unmarried']
race = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
sex = ['Female', 'Male']
nativeCountry = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico',
                 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']


def getIndex(array, value):
    for i, name in enumerate(array):
        if name == value.strip():
            return i
    return -1


def indexConvert(X):
    for i, values in enumerate(X):
        X[i][1] = getIndex(workclass, values[1])  # Workclass
        X[i][3] = getIndex(education, values[3])  # Education
        X[i][5] = getIndex(maritalStatus, values[5])  # Marital Status
        X[i][6] = getIndex(occupation, values[6])  # Occupation
        X[i][7] = getIndex(relationship, values[7])  # Relationship
        X[i][8] = getIndex(race, values[8])  # Race
        X[i][9] = getIndex(sex, values[9])  # Sex
        X[i][13] = getIndex(nativeCountry, values[13])  # native country
    return X


def indexRevert(X):
    converted = []

    converted.append(X[0])  # age
    converted.append(workclass[X[1]])  # workclass
    converted.append(X[2])  # fnlwgt
    converted.append(education[X[3]])  # education
    converted.append(X[4])  # education-enum
    converted.append(maritalStatus[X[5]])  # marital status
    converted.append(occupation[X[6]])  # occupation
    converted.append(relationship[X[7]])  # relationship
    converted.append(race[X[8]])  # race
    converted.append(sex[X[9]])  # sex
    converted.append(X[10])  # capital gain
    converted.append(X[11])  # capital loss
    converted.append(X[12])  # hours per week
    converted.append(nativeCountry[X[13]])  # native country

    return converted
