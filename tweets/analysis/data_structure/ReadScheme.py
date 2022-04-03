import re

accepted_words = ['rain','snow','patience','patient','sad','conges','jam','delay','stop','slow','block','wait', 'queu', 'flood','hard','abnormal',
                      'clos','stuffed','lock','roadwk','full','crook','heavy','obstruct','busy','stationary','standstill',
                      'busi','heavi','shut','accident','incident','slip','trap','divert','overturn','spillage','crash','crane',
                      'explosion','fire','burn','tch','lift','extinguish','stuck','breakdown','roll','damage','down','break','broken',
                      'broke','abnmal','fallen','debris','repair','disrupt','collide','collision','injuries','injury','ambulance','smoke',
                      'pain','emergency','police','officer','investigat','work','run','barrier','problem','trouble','issu','warn','caution']

directions_words = ['clockwise', 'anti-clockwise', 'anticlockwise']

connection_words = ['between', 'btwn']
junctions_words = ['j' + str(i) for i in range(2, 32)] + ['j1A']

class ReadScheme:

    def __init__(self):
        self.complete   = []
        self.contents   = None

        self.causes     = []
        self.junctions  = []
        self.directions = []
        self.connections= []

    def clear(self):
        self.causes.clear()
        self.junctions.clear()
        self.directions.clear()
        self.connections.clear()

    def addContent(self, content):
        self.contents = content

    def extractSchemes(self):
        complete    = False
        scheme  = ''

        content = re.sub(r'[^\w-]', ' ', self.contents)
        words = list(set(content.lower().split()))
        for word in words:
            for accepted_word in accepted_words:
                if word[:len(accepted_word)] == accepted_word:
                    self.causes.append(accepted_word)
                    scheme += accepted_word + ' '
                    break
            for junction in junctions_words:
                if word.lower() == junction:
                    self.junctions.append(junction)
                    scheme += junction + ' '
                    break
            for direction in directions_words:
                if word.lower() == direction:
                    self.directions.append(direction)
                    scheme += direction + ' '
                    break
            for connection in connection_words:
                if word.lower() == connection:
                    self.connections.append(connection)
                    scheme += connection + ' '
                    break

        if len(self.causes) > 0 and len(self.junctions) > 0 and len(self.directions) > 0:
            self.complete = True
        else:

            print('Less than junctions.')

        return self.complete, self.causes, self.directions, self.connections, self.junctions