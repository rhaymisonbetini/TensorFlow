$('#result').text('');

const tensorX = tf.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
const tensorY = tf.tensor([10, 20, 30, 40]);

const exibir = async (str = '') => {
	$('#result').text(str);
}

async function executar() {
	let text = '';
	let vetorX = await tensorToArray(tensorX);
	let vetorY = await tensorToArray(tensorY);

	let tamX = vetorX.length;
	let tamY = vetorY.length;

	let tempX = vetorX.slice(0, tamY);
	let tempY = vetorY;

	let dif = tamX - tamY;

	if (dif > 0) {
		let regressao = [];

		for (let i = 0; i < dif; i++) {
			let temporary =  await regressaoLinear(tempX, tempY, vetorX[tamY + i]);
			console.log(temporary);
			regressao.push(temporary);
		}

		let novoY = tempY.concat(regressao);
		let tensorZ = tf.tensor(novoY);

		text += 'ANTES \n';
		text += tensorX.toString(); + '\n\n';
		text += tensorY.toString(); +'\n\n';
		text += 'DEPOIS \n\n';
		text += tensorX.toString(); + '\n\n';
		text += tensorZ.toString(); +'\n\n';

	}

	exibir(text);
}


async function tensorToArray(tensor) {
	let array = [];
	let srtTenso = tensor.toString().replace('Tensor', '').trim();
	eval('array = ' + srtTenso);
	return array;
}

async function arrayToTensor(array) {
	let tensor = tf.tensor(array);
	return tensor;
}

async function regressaoLinear(arrayX, arrayY, pattern) {

	let x = await arrayToTensor(arrayX);
	let y = await arrayToTensor(arrayY);

	//formula da regressao linear simples
	let resultado1 = x.sum().mul(y.sum()).div(x.size1);
	let resultado2 = x.sum().mul(x.sum()).div(x.size);
	let resultado3 = x.mul(y).sum().sub(resultado1);
	let resultado4 = resultado3.div(x.square().sum().sub(resultado2));
	let resultado5 = y.mean().sub(resultado4.mul(x.mean()));

	let tensor = resultado4.mul(pattern).add(resultado5);
	let array = await tensorToArray(tensor);
	return array;

} 
