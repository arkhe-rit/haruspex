
const arrayContainsAll = (subArray) => (arr) => {
  return subArray.every((subArrayItem) => arr.includes(subArrayItem));
};

const arrayLevenshteinDistance = (arr1, arr2) => {
  // Inserts and deletes are weighted equally as 1
  // Substitutions are weighted as 2
  const matrix = [];

  for (let i = 0; i <= arr1.length; i++) {
    matrix[i] = [i];
  }
  for (let j = 0; j <= arr2.length; j++) {
    matrix[0][j] = j;
  }

  for (let i = 1; i <= arr1.length; i++) {
    for (let j = 1; j <= arr2.length; j++) {
      const substitutionCost = arr1[i - 1] === arr2[j - 1] ? 0 : 2;
      matrix[i][j] = Math.min(
        matrix[i - 1][j] + 1, // deletion
        matrix[i][j - 1] + 1, // insertion
        matrix[i - 1][j - 1] + substitutionCost // substitution
      );
    }
  }

  return matrix[arr1.length][arr2.length];
};

const arrayMode = (arrays) => {
  // `arrays` is an array of arrays of strings
  // The mode of `arrays` is the array of strings whose average Levenshtein distance to the other arrays is the lowest

  // First, we need to find the average Levenshtein distance between each array and the other arrays
  const averageDistances = arrays.map((arr1, i) => {
    const otherArrays = arrays.filter((_, j) => i !== j);
    const distances = otherArrays.map((arr2) =>
      arrayLevenshteinDistance(arr1, arr2)
    );
    return distances.reduce((a, b) => a + b) / distances.length;
  });

  // Then, we return the array with the lowest average distance
  return arrays[averageDistances.indexOf(Math.min(...averageDistances))];
}

const pick = (array) => array[Math.floor(Math.random() * array.length)]

export {
  arrayLevenshteinDistance,
  arrayContainsAll,
  arrayMode,
  pick
}