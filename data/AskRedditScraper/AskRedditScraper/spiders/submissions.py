import scrapy, math, json

questionFileSuffix = "-question.txt"
commentFileSuffix = "-comment.txt"
scoreFileSuffix = "-score.txt"
subreddit = 'AskReddit'

endDate = 3650
messageByPeriod = 500
targetNumber =  50000

class SubmissionsSpider(scrapy.Spider):
    name = "submissions"

    def start_requests(self):
        periodSize = int(endDate * messageByPeriod / targetNumber)
        numberOfPeriods = math.ceil(3650/periodSize)
        periods = [str(i) for i in range(0, endDate, periodSize)]

        urls = []
        periods_url = []

        for period in periods:
            before = ''
            after = ''
            if periods.index(period) == (len(periods) - 1):
                before = period + 'd'
            else:
                after = periods[periods.index(period) + 1] + 'd'
                before = period + 'd'

            urls.append('https://api.pushshift.io/reddit/search/submission/?size={}&subreddit={}&sort_type=score&before={}&after={}&score=>5&sort=desc'.format(str(messageByPeriod), subreddit, before, after))
            periods_url.append(period)

        for i in range(len(urls)):
            yield scrapy.Request(url=urls[i], callback=self.parseSubmission, meta={'period': periods_url[i]})

    def parseSubmission(self, response):
        jsonresponse = json.loads(response.body_as_unicode())
        ids = []
        scores = []
        urls = []
        titles = []
        treated_ids = []
        period = response.meta['period']

        with open(period + scoreFileSuffix, 'r', encoding='utf-8') as c:
            content = c.readlines()

        treated_ids = [x.split(' ')[0] for x in content]

        for post in jsonresponse['data']:
            post_id = post["id"]
            if not(post_id in treated_ids):
                ids.append(post_id)
                scores.append(post["score"])
                titles.append(post["title"])
                urls.append(post["url"] + ".json")

        for i in range(len(urls)):
            yield scrapy.Request(url=urls[i], callback=self.parseComment,
                meta={'period': response.meta['period'],
                      'id': ids[i],
                      'score': scores[i],
                      'title': titles[i]}
                )

    def parseComment(self, response):
        comments = json.loads(response.body_as_unicode())
        comments.pop(0)
        answer = comments[0]['data']['children'][0]['data']['body']
        j=1
        while "I am a bot" in answer or "[removed]" == answer:
            answer = comments[0]['data']['children'][j]['data']['body']
            j+=1
        answer = answer.replace('\n', ' ')
        period = response.meta['period']
        id_post = response.meta['id']
        score = response.meta['score']
        title = response.meta['title']
        with open(period + commentFileSuffix, 'a', encoding='utf-8') as c, open(period + questionFileSuffix, 'a', encoding='utf-8') as q, open(period + scoreFileSuffix, 'a', encoding='utf-8') as s:
            c.writelines(id_post + " " + answer)
            c.writelines('\n')
            q.writelines(id_post + " " + title)
            q.writelines('\n')
            s.writelines(id_post + " " + str(score))
            s.writelines('\n')
