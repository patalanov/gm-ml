


def column_tempo(lance):
    if 'S' not in lance:
        lance = lance+'00'
    if 'M' in lance: 
        if len(lance.split('M')[0].split('T')[1])==1:
            tempo = '0'+lance.split('M')[0].split('T')[1]+':'+lance.split('M')[1].replace('S','')
        else:
            tempo = lance.split('M')[0].split('T')[1]+':'+lance.split('M')[1].replace('S','')
    else:
        tempo = lance.replace('PT','00:').replace('S','')
    
    if len(tempo.split(':')[1])==1:
        tempo = tempo.split(':')[0]+':0'+tempo.split(':')[1]
    return tempo



