# EXERCISE: TRANSLATOR

from translate import Translator

translator = Translator(to_lang = 'ja')

# gubbins.txt file is a local file in directory, to import any file, perhaps on desktop
# use ./gubbins.txt - universal file calling on current machine or machine environment?

try:
    with open('./gubbins.txt', mode = 'r') as my_file:
        text = my_file.read()
        print(text)
        translation = translator.translate(text)
        print(translation)
        with open('gubbins_japanese.txt', mode = 'w', encoding = 'utf-8') as jap_file:      # make sure encoding is set for this
            jap_file.write(translation)       
except FileNotFoundError as e:
    print('check ur fgile pathg')