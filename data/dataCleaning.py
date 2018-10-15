from os import listdir
from os.path import isfile, join

path = "askreddit/"
outputFile = "askredditData.txt"

class Period():
    def __init__(self, period, questionPath, commentPath, scorePath):
        self.questionPath = questionPath
        self.commentPath = commentPath
        self.scorePath = scorePath
        self.period = period

    def getCleanedPeriod(self):

        nbPost = 0
        output = ""
        with open(path + self.questionPath, 'r') as q, open(path + self.commentPath, 'r') as c, open(path + self.scorePath, 'r') as s:
            questions = q.readlines()
            comments = c.readlines()
            scores = s.readlines()

        i_comment = 0
        i_question = 0

        postId = []
        postScores = []

        for i in range(len(scores)):
            splitted = scores[i].split(' ')
            postId.append(splitted[0])
            postScores.append(splitted[1].replace('\n', ''))

        scores = postScores

        for i in range(len(postId)):
            if i < len(postId) - 1:
                question = questions[i_question].replace(str(postId[i]) + ' ', '')
                comment = comments[i_comment].replace(str(postId[i]) + ' ', '')
                i_question += 1
                i_comment += 1
                while not(postId[i+1] in questions[i_question]):
                    question += questions[i_question].replace('\n', '')
                    i_question += 1
                while not(postId[i+1] in comments[i_comment]):
                    comment += comments[i_comment].replace('\n', '')
                    i_comment += 1
            else:
                question = questions[i_question].replace(str(postId[i]) + ' ', '')
                comment = comments[i_comment].replace(str(postId[i]) + ' ', '')
                i_question += 1
                i_comment += 1
                while i_question < len(questions):
                    question += questions[i_question]
                    i_question += 1
                while i_comment < len(comments):
                    comment += comments[i_comment]
                    i_comment += 1

            comment = comment.lower()
            question = question.lower()
            TAG = ["(nsfw)", "[nsfw]", "(serious)", "[serious]", "[megathread]", "(megathread)"]
            for tag in TAG:
                question = question.replace(tag, '')
                comment = comment.replace(tag, '')
            if len(comment.split(' ')) <= 50:
                question = question.replace('\n', '').replace("'", '').replace('"', '').replace(',', ' , ').replace('!', ' ! ').replace('?', ' ? ').replace('.', ' . ').replace(';', ' ; ').replace('(', ' ( ').replace(')', ' ) ').replace('[', ' [ ').replace(']', ' ] ').replace('{', ' { ').replace('}', ' } ').replace('1 , 0', '1,0').replace('0 , 0', '0,0').replace('.  .', '..').replace('!  !', '!!').replace(':', ' : ').replace('/', ' / ').replace(' nt', " 'nt").replace("-", " - ")
                comment = comment.replace('\n', '').replace("'", '').replace('"', '').replace(',', ' , ').replace('!', ' ! ').replace('?', ' ? ').replace('.', ' . ').replace(';', ' ; ').replace('(', ' ( ').replace(')', ' ) ').replace('[', ' [ ').replace(']', ' ] ').replace('{', ' { ').replace('}', ' } ').replace('1 , 0', '1,0').replace('0 , 0', '0,0').replace('.  .', '..').replace('!  !', '!!').replace(':', ' : ').replace('/', ' / ').replace(' nt', " 'nt").replace("-", " - ")
                question = question.replace("/ r /", '').replace("..  .", "...").replace("!!  !", "!!!").replace("&", " & ").replace("$", " $ ").replace("*", " * ").replace('%', ' * ').replace("! !", "!!").replace("?  ?", "??").replace("? ?", "??")
                comment = comment.replace("/ r /", '').replace("..  .", "...").replace("!!  !", "!!!").replace("&", " & ").replace("$", " $ ").replace("*", " * ").replace('%', ' * ').replace("! !", "!!").replace("?  ?", "??").replace("? ?", "??")
                if not("http" in comment) and not("Http" in comment) and not("[ deleted ]" in comment) and not("[ Deleted ]" in comment):
                    nbPost += 1
                    output += str(postId[i]) + ' ' + str(scores[i]) + '\n' + question + '\n' + comment + '\n' + '\n'

        return output, nbPost

    def getPeriod(self):
        return self.period


def getFiles():
    periods = []
    scores = []
    comments = []
    questions = []
    files = [f for f in listdir(path) if isfile(join(path, f))]

    for f in files:
        try:
            if "comment" in f:
                comments.append(f)
            elif "question" in f:
                questions.append(f)
            elif "score" in f:
                scores.append(f)
            f = f.split('-')
            if not(int(f[0]) in periods):
                periods.append(int(f[0]))
        except:
            print("File with bad format: " + f[0])

    periods.sort()
    scores.sort(key=lambda x:int(x.split('-')[0]))
    questions.sort(key=lambda x:int(x.split('-')[0]))
    comments.sort(key=lambda x:int(x.split('-')[0]))

    objectList = []

    for i in range(len(periods)):
        if periods[i] == int(scores[i].split('-')[0]) == int(comments[i].split('-')[0]) == int(questions[i].split('-')[0]):
            objectList.append(Period(periods[i], questions[i], comments[i], scores[i]))
        else:
            print("Error with period {}, one file might be missing.".format(periods[i]))

    return objectList


periods = getFiles()
output = ""
nbPost = 0
for period in periods:
    s, nb = period.getCleanedPeriod()
    print("Cleaning period {}.".format(period.getPeriod()))
    output += s
    nbPost += nb

print("Writing in file...")

with open(outputFile, 'w') as f:
    f.writelines(output)

print("{} remaining Q&A after cleaning.".format(nbPost))
print("Cleaning done.")
