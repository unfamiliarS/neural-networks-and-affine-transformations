# Neural Networks & Affine Transformations
A research project exploring the application of affine transformations to fully connected neural networks, implemented in Java. Three types of transformations were considered and tested: rotation, sheare, scaling.

## Examples
<img width="2560" height="1329" alt="rotate" src="https://github.com/user-attachments/assets/86b25dd6-e036-4c98-8e87-980089223354" />
<img width="2560" height="1329" alt="shear" src="https://github.com/user-attachments/assets/d435919c-d538-47dd-86c3-4bf5e1195299" />

## Project Structure
```
neural-networks-and-affine-transformations/
├── src/             # Source code
├── docs/            # Documentation files (explanatory note and presentation)
├── results/         # Resulting experiments and images 
├── assembly/        # Build configuration
├── affine-network   # Launch script (see building section below)
```

## Building, installing and usage
For building from source in project root execute
```
mvn clean package
```
From src/main/resources copy needed models into $HOME/.local/affine
```
mkdir -p $HOME/.local/affine
cp -r src/main/resources/* $HOME/.local/affine/
```
Use affine-network script
```
chmod +x affine-network
./affine-network
```
