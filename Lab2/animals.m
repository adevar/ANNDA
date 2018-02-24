%4.2

%globals
global epochs eta radius numAnimals numAttributes numWeights;
epochs = 20;
eta = 0.2;
radius = 25;
numAnimals = 32;
numAttributes = 84;
numWeights = 100;

%import input data
%cities = readtable('cities.dat','Delimiter', ';', 'ReadVariableNames', false)
animals = csvread('animals.dat');
animals = reshape(animals, [numAnimals, numAttributes]);
names = importdata('animalnames.txt');

%create weight matrix
%weights = 0.5 * ones(numWeights, numAttributes);
weights = rand(numWeights, numAttributes);

for i = 1:epochs
	for j = 1:numAnimals

% Calculate the similarity between the input pattern and the weights arriving at each output node.
% Find the most similar node; often referred to as the winner.
		winner = mostSimilar(animals(j,:), weights);
		%disp(winner);

% Update the weights of all nodes in the weight matrix (neighbourhood) such that their
% weights are proportionally moved closer to the input pattern.
		weights = updateWeights(i, winner, weights, animals(j, :));
	end
end

% 32 element vector with the indices of the winning nodes for each animal
pos = zeros(numAnimals, 1);
for i = 1:numAnimals
	pos(i) = mostSimilar(animals(i,:), weights);
end

% sort the vector
A = array2table(pos, 'RowNames', names);
ordered = sortrows(A);

% local function definitions
function closest = mostSimilar(x, W)
	similarity = zeros(size(W,1), 1);
	for i = 1:size(W,1)
  	  diff = x - W(i,:);
      similarity(i) = norm(diff); %length of difference vector
	end
	[~, closest] = min(similarity); % index of most similar node
end

function sigma = mySigma(epoch)
    global radius epochs;
	timeConstant = -(epochs - 1)/log(1/radius);
	sigma = radius*exp(-(epoch-1)/timeConstant);
    %sigma = radius - (radius/(epochs-1))*(epoch-1);
end

function neighborlyRate = neighbors(curr, epoch, winner)
  %global epochs radius;
  sigma = mySigma(epoch);
  d = winner - curr;
  if (d^2 < sigma^2)
    neighborlyRate = exp(-(d^2/(2*(sigma^2)))); 
    %neighborlyRate = 1;
  else
    neighborlyRate = 0;
  end
end

function weights = updateWeights(epoch, winner, weights, animal)
  global eta numWeights;
  for w = 1:numWeights
      rate = neighbors(w, epoch, winner);
      weights(w, :) = weights(w, :) + rate * eta * (animal - weights(w, :));
  end
end