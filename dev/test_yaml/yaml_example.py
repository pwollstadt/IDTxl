import yaml

# read yaml from file
f = open("yaml_ex.yaml",'r')
my_dict1 = yaml.load(f)
f.close()

my_dict2 = yaml.load("""
... name: Vorlin Laruknuzum
... sex: Male
... class: Priest
... title: Acolyte
... hp: [32, 71]
... sp: [1, 13]
... gold: 423
... inventory:
... - a Holy Book of Prayers (Words of Wisdom)
... - an Azure Potion of Cure Light Wounds
... - a Silver Wand of Wonder
... """)

# write to file (this is not so pretty, don't know why)
f = open("my_dump.yaml", 'w')
for k in my_dict2
	f.write(yaml.dump(my_dict2[k], default_flow_style=False, allow_unicode=True))
f.close()