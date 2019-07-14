# Pythono3 code to rename multiple 
# files in a directory or folder 

# importing os module 
import os 

# Function to rename multiple files 
def main(): 
	src_dir ="F:\\PycharmProjects\\ai_Lessons\\img"
	j = 0
	for dirname in os.listdir(src_dir): 
		i = 1
		for filename in os.listdir(src_dir + "\\" + dirname):
			dst ="img" + f'{j:02}' + "-" + f'{i:08}' + ".png"
			src = src_dir + "\\" + dirname + "\\" + filename 
			dst = src_dir + "\\" + dirname + "\\" + dst
			print("Directory --> '" + dirname + "' , File Number " + f'{i:08}' + " out of " + str(len(os.listdir(src_dir + "\\" + dirname))))
			# rename() function will 
			# rename all the files 
			os.rename(src, dst) 
			i += 1
		j += 1

# Driver Code 
if __name__ == '__main__': 
	
	# Calling main() function 
	main() 
