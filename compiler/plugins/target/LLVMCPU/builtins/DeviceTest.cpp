// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/LLVMCPU/builtins/Device.h"

#include "gtest/gtest.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"

namespace mlir::iree_compiler::IREE::HAL {
namespace {

// Bitcode selection and specialization only need the target triple/features.
class TestTargetMachine : public llvm::TargetMachine {
public:
  TestTargetMachine(const llvm::Target &target, StringRef triple)
      : TargetMachine(target, "", llvm::Triple(triple), "", "",
                      llvm::TargetOptions()) {}
};

static void checkFloat16Builtins(StringRef triple, bool isWasm) {
  SCOPED_TRACE(triple.str());
  llvm::Target target;
  TestTargetMachine targetMachine(target, triple);
  llvm::LLVMContext context;
  auto moduleOr = loadDeviceBitcode(&targetMachine, context);
  ASSERT_TRUE(static_cast<bool>(moduleOr))
      << llvm::toString(moduleOr.takeError());
  auto module = std::move(*moduleOr);
  module->setTargetTriple(targetMachine.getTargetTriple());
  specializeDeviceModule({}, *module, targetMachine);
  EXPECT_FALSE(llvm::verifyModule(*module, &llvm::errs()));

  auto *f32Type = llvm::Type::getFloatTy(context);
  auto *f64Type = llvm::Type::getDoubleTy(context);
  llvm::Type *halfType = isWasm ? llvm::Type::getInt16Ty(context) : f32Type;
  struct {
    StringRef name;
    llvm::Type *result;
    llvm::Type *argument;
  } expected[] = {
      {"__extendhfsf2", f32Type, halfType},
      {"__extendhfdf2", f64Type, halfType},
      {"__truncsfhf2", halfType, f32Type},
      {"__truncdfhf2", halfType, f64Type},
  };
  for (auto &builtin : expected) {
    auto *function = module->getFunction(builtin.name);
    ASSERT_NE(function, nullptr);
    EXPECT_EQ(
        function->getFunctionType(),
        llvm::FunctionType::get(builtin.result, {builtin.argument}, false))
        << builtin.name.str();
  }
}

TEST(DeviceTest, WasmFloat16Builtins) {
  checkFloat16Builtins("wasm32-unknown-unknown", true);
  checkFloat16Builtins("wasm64-unknown-unknown", true);
}

TEST(DeviceTest, NativeFloat16Builtins) {
  checkFloat16Builtins("x86_64-unknown-linux-gnu", false);
  checkFloat16Builtins("aarch64-unknown-linux-gnu", false);
  checkFloat16Builtins("i386-unknown-linux-gnu", false);
}

} // namespace
} // namespace mlir::iree_compiler::IREE::HAL
