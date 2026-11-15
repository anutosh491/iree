// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/LLVMCPU/builtins/Device.h"

#include "iree/builtins/device/libdevice_bitcode.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "mlir/Support/LLVM.h"

namespace mlir::iree_compiler::IREE::HAL {

static const iree_file_toc_t *lookupDeviceFile(StringRef filename) {
  for (size_t i = 0; i < iree_builtins_libdevice_bitcode_size(); ++i) {
    const auto &file_toc = iree_builtins_libdevice_bitcode_create()[i];
    if (filename == file_toc.name) {
      return &file_toc;
    }
  }
  return nullptr;
}

static const iree_file_toc_t *
lookupDeviceFile(llvm::TargetMachine *targetMachine) {
  const auto &triple = targetMachine->getTargetTriple();

  // NOTE: other arch-specific checks go here.

  if (triple.isWasm()) {
    // TODO(benvanik): feature detect simd and such.
    // auto features = targetMachine->getTargetFeatureString();
    if (triple.isArch32Bit()) {
      return lookupDeviceFile("libdevice_wasm32_generic.bc");
    } else if (triple.isArch64Bit()) {
      return lookupDeviceFile("libdevice_wasm64_generic.bc");
    }
  }

  // Fallback path using the generic wasm variants as they are largely
  // machine-agnostic.
  if (triple.isArch32Bit()) {
    return lookupDeviceFile("libdevice_wasm32_generic.bc");
  } else if (triple.isArch64Bit() &&
             targetMachine->getTargetFeatureString().contains("+sme")) {
    return lookupDeviceFile("libdevice_aarch64_sme.bc");
  } else if (triple.isArch64Bit()) {
    return lookupDeviceFile("libdevice_wasm64_generic.bc");
  } else {
    return nullptr;
  }
}

llvm::Expected<std::unique_ptr<llvm::Module>>
loadDeviceBitcode(llvm::TargetMachine *targetMachine,
                  llvm::LLVMContext &context) {
  // Find a bitcode file for the current architecture.
  const auto *file = lookupDeviceFile(targetMachine);
  if (!file) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "no matching architecture bitcode file");
  }

  // Load the generic bitcode file contents.
  llvm::MemoryBufferRef bitcodeBufferRef(
      llvm::StringRef(file->data, file->size), file->name);
  auto bitcodeModuleValue = llvm::parseBitcodeFile(bitcodeBufferRef, context);
  if (!bitcodeModuleValue) {
    return bitcodeModuleValue;
  }
  auto bitcodeModule = std::move(bitcodeModuleValue.get());

  // Clang adds its own per-function attributes that we need to strip so that
  // our current executable variant target is used instead.
  for (auto &func : bitcodeModule->functions()) {
    func.removeFnAttr("target-cpu");
    func.removeFnAttr("tune-cpu");
    func.removeFnAttr("target-features");
  }

  return std::move(bitcodeModule);
}

static void overridePlatformGlobal(llvm::Module &module, StringRef globalName,
                                   uint32_t newValue) {
  // NOTE: the global will not be defined if it is not used in the module.
  auto *globalValue = module.getNamedGlobal(globalName);
  if (!globalValue) {
    return;
  }
  globalValue->setLinkage(llvm::GlobalValue::PrivateLinkage);
  globalValue->setDSOLocal(true);
  globalValue->setConstant(true);
  globalValue->setInitializer(
      llvm::ConstantInt::get(globalValue->getValueType(), APInt(32, newValue)));
}

static void specializeWasmFloat16Builtins(llvm::Module &module) {
  auto &context = module.getContext();
  auto *i16Type = llvm::Type::getInt16Ty(context);
  auto *f32Type = llvm::Type::getFloatTy(context);
  auto *f64Type = llvm::Type::getDoubleTy(context);
  struct {
    StringRef name;
    llvm::Type *result;
    llvm::Type *argument;
  } builtins[] = {
      {"__extendhfsf2", f32Type, i16Type},
      {"__extendhfdf2", f64Type, i16Type},
      {"__truncsfhf2", i16Type, f32Type},
      {"__truncdfhf2", i16Type, f64Type},
  };

  // Drop the old bodies together, as the double helpers call the float helpers.
  for (auto &builtin : builtins) {
    module.getFunction(builtin.name)->dropAllReferences();
  }
  for (auto &builtin : builtins) {
    module.getFunction(builtin.name)->eraseFromParent();
    auto *function = llvm::Function::Create(
        llvm::FunctionType::get(builtin.result, {builtin.argument}, false),
        llvm::GlobalValue::ExternalLinkage, builtin.name, module);
    llvm::IRBuilder<> builder(
        llvm::BasicBlock::Create(context, "entry", function));
    llvm::Value *argument = function->getArg(0);
    if (builtin.argument == f64Type) {
      argument = builder.CreateFPTrunc(argument, f32Type);
    }
    auto *conversion = module.getFunction(
        builtin.argument == i16Type ? "iree_h2f_ieee" : "iree_f2h_ieee");
    llvm::Value *result = builder.CreateCall(conversion, {argument});
    if (builtin.result == f64Type) {
      result = builder.CreateFPExt(result, f64Type);
    }
    builder.CreateRet(result);
  }
}

void specializeDeviceModule(IREE::HAL::ExecutableVariantOp variantOp,
                            llvm::Module &module,
                            llvm::TargetMachine &targetMachine) {
  overridePlatformGlobal(module, "libdevice_platform_example_flag", 0u);
  // Wasm passes half bits as integers. Select this ABI for the final target,
  // since the same bitcode is also used as a fallback for native CPUs.
  if (targetMachine.getTargetTriple().isWasm()) {
    specializeWasmFloat16Builtins(module);
  }
}

} // namespace mlir::iree_compiler::IREE::HAL
