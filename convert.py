import re
def chatbot_program(code = "", func_name=""):
    first_line = "'use strict';\n"
    start_start = "\nconst start = (say, sendButton) => {\n"
    start_end = "\n};"
    state = ''
    state_start = "\nconst state = (payload, say, sendButton) => {\nlet ary = payload.split('-');\n"
    state_end = "\n};"
    if func_name == "":
        filename='func'
        title = filename
        introduction = ''
        code = code.replace('function','function func')
        code = code.replace(filename+' ',filename)
        params = []
        if re.search('(?<=function '+filename+').*?(?=\{)',code) != None:
            params = re.search('(?<=function '+filename+').*?(?=\{)',code).group(0).replace('(','').replace(')','').split(',')
        start = start_start +"sendButton('You need to input "+str(len(params))+" parameter(s) for "+ str(filename) +", [{title: 'OK', payload: '1'}]);"+ start_end
        for i in range(len(params)):
            state_start = state_start + "let "+params[i]+" = ary["+str(i)+"];\n"
        caller = func_name +'('
        for i in range(len(params)):
            caller+=params[i]
            if i!=len(params)-1:
                caller+=','
        caller += ')' 
        state = '\n'+code + state_start+ caller +state_end
        file_setting = "\nmodule.exports = {\nfilename:'" +filename+"',\ntitle: '"+title+"',\nintroduction: ["+introduction+"],\nstart: start,\nstate: state\n};"
        converted_file = first_line + start + state + file_setting
        return converted_file
    elif code.find('function ')!=-1:
        filename = func_name
        title = filename
        introduction = ''
        params = []
        if re.search('(?<=function '+filename+').*?(?=\{)',code) != None:
            params = re.search('(?<=function '+filename+').*?(?=\{)',code).group(0).replace('(','').replace(')','').split(',')
        start = start_start +"sendButton('You need to input "+str(len(params))+" parameter(s) for "+ str(filename) +", [{title: 'OK', payload: '1'}]);"+ start_end
        for i in range(len(params)):
            state_start = state_start + "let "+params[i]+" = ary["+str(i)+"];\n"
        # caller = re.search('(?<=function ).*?(?=\{)',code).group(0)
        caller = func_name +'('
        for i in range(len(params)):
            caller+=params[i]
            if i!=len(params)-1:
                caller+=','
        caller += ')'
        state_start = state_start + caller +';'
        state = state_start +state_end
        file_setting = "\nmodule.exports = {\nfilename:'" +filename+"',\ntitle: '"+title+"',\nintroduction: ["+introduction+"],\nstart: start,\nstate: state\n};"
        converted_file = first_line + start + '\n'+code + state + file_setting
        return converted_file
    else:
        filename='func'
        title = filename
        introduction = ''
        start = start_start +"sendButton('You need to input 0 parameter(s) for "+ str(filename) +", [{title: 'OK', payload: '1'}]);"+ start_end
        state = state_start+ code +state_end
        file_setting = "\nmodule.exports = {\nfilename:'" +filename+"',\ntitle: '"+title+"',\nintroduction: ["+introduction+"],\nstart: start,\nstate: state\n};"
        converted_file = first_line + start + state + file_setting
        return converted_file