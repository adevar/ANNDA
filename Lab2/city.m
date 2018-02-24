%4.2

%globals
global epochs eta radius numCities numAttributes numWeights;
epochs = 20;
eta = 0.2;
radius = 1;
numCities = 10;
numAttributes = 2;
numWeights = 10;

%import input data
%cities = readtable('cities.dat','Delimiter', ';', 'ReadVariableNames', false)
cities = csvread('cities.dat');
cities = reshape(cities, [numCities, numAttributes]);

%create weight matrix
weights = 0.5 * ones(numWeights, numAttributes);
%weights = rand(numWeights, numAttributes);
init = weights;

for i = 1:epochs
	for j = 1:numCities

% Calculate the similarity between the input pattern and the weights arriving at each output node.
% Find the most similar node; often referred to as the winner.
		winner = mostSimilar(cities(j,:), weights);
		%disp(winner);

% Update the weights of all nodes in the weight matrix (neighbourhood) such that their
% weights are proportionally moved closer to the input pattern.
		weights = updateWeights(i, winner, weights, cities(j, :));

	end

%   tour = zeros(numCities, 1);
%   for i = 1:numCities
%     tour(i) = mostSimilar(cities(i,:), weights);
%   end
% 
%   A = horzcat(cities, tour);
%   ordered = sortrows(horzcat(cities,tour),3);
%   %A = array2table(pos, 'RowNames', names);
%   %ordered = sortrows(A);
%   ordered = [ordered; ordered(1,:)];
%   plot(ordered(:,1), ordered(:,2), '-x', weights(:,1), weights(:,2), 'r');

end

% 32 element vector with the indices of the winning nodes for each animal
tour = zeros(numCities, 1);
for i = 1:numCities
	tour(i) = mostSimilar(cities(i,:), weights);
end

A = horzcat(cities, tour);
ordered = sortrows(horzcat(cities,tour),3);
%A = array2table(pos, 'RowNames', names);
%ordered = sortrows(A);
ordered = [ordered; ordered(1,:)];
plot(ordered(:,1), ordered(:,2), '-x')

distance = 0;
for i = 1:(size(ordered) - 1)
    distance = distance + norm(ordered(i,1:2) - ordered(i + 1,1:2));
end
disp(distance);

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
	timeConstant = -(epochs - 1)/log(.01/radius);
	sigma = radius*exp(-(epoch-1)/timeConstant);
end

function neighborlyRate = neighbors(curr, epoch, winner)
  %global epochs radius;
  sigma = mySigma(epoch);
  %sigma = radius - (radius/epochs)*(epoch-1);
  d = min(abs(winner - curr), mod(abs(abs(winner - curr) - 10), 10)) ;
  if (d^2 < sigma^2)
    %neighborlyRate = exp(-(d^2/(2*(sigma^2)))); 
    neighborlyRate = 1;
  else
    neighborlyRate = 0;
  end
end

function weights = updateWeights(epoch, winner, weights, city)
  global eta numWeights;
  for w = 1:numWeights
      rate = neighbors(w, epoch, winner);
      weights(w, :) = weights(w, :) + rate * eta * (city - weights(w, :));
  end
end