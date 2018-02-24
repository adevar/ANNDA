%4.2

%globals
global epochs eta radius numMps numAttributes dimWeights;
epochs = 20;
eta = 0.2;
radius = 10.0;
numMps = 349;
numAttributes = 31;
dimWeights = [10,10];

%import input data
votes = csvread('votes.dat');
votes = reshape(votes, [numMps, numAttributes]);
gender = csvread('mpsex.dat',2);
party = csvread('mpparty.dat',2);
district = csvread('mpdistrict.dat',2);

%create weight matrix
weights = rand(dimWeights(1), dimWeights(2), numAttributes);
%weights = 0.5*ones(dimWeights(1), dimWeights(2), numAttributes);
%init = weights;

for i = 1:epochs
	for j = 1:numMps
% Calculate the similarity between the input pattern and the weights arriving at each output node.
% Find the most similar node; often referred to as the winner.
		winner = mostSimilar(votes(j,:), weights);

% Update the weights of all nodes in the weight matrix (neighbourhood) such that their
% weights are proportionally moved closer to the input pattern.
		weights = updateWeights(i, winner, weights, votes(j, :));
    end
end

% 349 element vector with the indices of the winning [x,y] indices for each MP
indices = zeros(numMps, 2);
for i = 1:numMps
	indices(i, :) = mostSimilar(votes(i,:), weights);
end

figure
genderAxes = subplot(3, 1, 1);
genderColor = ["r*","b*"];
colorPlot(indices, gender, genderColor, 2, genderAxes);

partyAxes = subplot(3, 1, 2);
partyColor = ["r*", "b*","y*", "m*", "c*", "g*", "w*", "k*"];
colorPlot(indices, party, partyColor, 8, partyAxes);

% districtAxes = subplot(3, 1, 3);
% districtColor = []; % don't really want to hardcode 29 colors
% colorPlot(indices, gender, districtColor, 30, districtAxes); % district number goes from 1 to 29

% local function definitions
function colorPlot(indices, labels, colors, number, axes)
    hold on
    horzcat(labels, indices);
    for i = 0:(number - 1)
        p = (labels == i);
        plot(axes, indices(p, 1), indices(p, 2), colors(i + 1));
    end
    hold off
end

function closest = mostSimilar(x, W)
	similarity = zeros(size(W,1), size(W,2));
	for i = 1:size(W,1)
        for j = 1:size(W,2)
            diff = x - squeeze(W(i,j,:)).';
            similarity(i,j) = norm(diff); %length of difference vector
        end
	end
	[~, index] = min(similarity(:)); % index of most similar node
	[a,b] = ind2sub(size(similarity), index);
    closest = [a,b];
end

function sigma = mySigma(epoch)
    global radius epochs;
	timeConstant = -(epochs - 1)/log(.01/radius);
	sigma = radius * exp(-(epoch-1)/timeConstant);
    %sigma = radius - (radius/epochs)*(epoch-1);
end

function neighborlyRate = neighbors(curr, epoch, winner)
  sigma = mySigma(epoch);
  d = abs(winner(1) - curr(1)) + abs(winner(2) - curr(2));
  if (d < sigma)
    neighborlyRate = exp(-(d^2/(2*(sigma^2)))); 
    %neighborlyRate = 1;
  else
    neighborlyRate = 0;
  end
end

function weights = updateWeights(epoch, winner, weights, vote)
  global eta dimWeights;
  for x = 1:dimWeights(1)
      for y = 1:dimWeights(2)
          curr = [x, y];
          rate = neighbors(curr, epoch, winner);
          weights(x, y, :) = squeeze(weights(x, y, :)).' + rate * eta * (vote - squeeze(weights(x, y, :)).');
      end
  end
end