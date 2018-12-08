
### ---- Normalizing features ---- ###

def main():
	from sklearn.preprocessing import StandardScaler
	sc = StandardScaler()
	x_train = sc.fit_transform(X = x_train)
	x_test = sc.transform(X = x_test)

	print(x_train.shape, x_test.shape)

	print(x_train, x_test)
	print(type(x_train))
	print(type(x_test))
	#ssert(0)

	print("Normalized features")
	print(x_train, x_test)

	return x_train, x_test

if __name__ == '__main__':
	main()