import requests
print('Beginning file download with requests')
i=1250
notfound=0;
while notfound<=2000:
	url = 'http://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/00001.'+"{:05d}".format(i)+'.mp4'
	print('looking for 00001.'+"{:05d}".format(i)+'.mp4')
	r = requests.get(url)
	if r.status_code==404:
		print("Can't find video")
		notfound=notfound+1;
	else:
		with open('video'+"{:05d}".format(i)+'0.mp4', 'wb') as f:
		    f.write(r.content)

		# Retrieve HTTP meta-data
		print(r.status_code)
		print(r.headers['content-type'])
		print(r.encoding)
		notfound=0;
	i=i+1
